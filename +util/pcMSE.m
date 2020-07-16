function pcMSE(modelfile, datafile)
% mean squared error on power/coherence estimates from CSFA model

load(modelfile, 'sets', 'csfaModels','labels')
load(datafile, 'power', 'coherence', 'xFft')

for m = 1:numel(csfaModels)
    
    %% get train model UKU
    model = csfaModels{m}.trainModels(end);
    if isa(model, 'GP.dCSFA')
        model = model.kernel;
    end
    
    freqIdx = model.freqBand(labels.f);
    f = labels.f(freqIdx);
    
    W = sum(model.W);
    C = model.C;
    F = length(f);
    % compile UKU from each window
    ampUKU = abs(model.UKU(f, [1:W]));
    
    %% repeat for holdout model UKU
    if isfield(csfaModels{m}, 'holdoutModels')
        model = csfaModels{m}.holdoutModels(end);
        
        W2 = sum(sets.val);
        % compile UKU from each validation window
        valWindows = find(sets.val(~sets.train));
        ampUKU2 = abs(model.UKU(f, valWindows));
        holdout = true;
    else
        holdout = false;
    end
    
    %% calculate normalized power (to match what we fit in CSFA)
    if m==1
        normConst = mean(abs(xFft).^2,3);
        
        % find normConst values corresponding to f(requency) values
        % by averaging normConst values corresponding to adjacent frequencies
        newConst = zeros(length(f),C);
        for k = 1:length(f)
            lowFConst = normConst(find(labels.s < f(k), 1, 'last'),:);
            highFConst = normConst(find(labels.s > f(k), 1, 'first'),:);
            newConst(k,:) = mean([lowFConst; highFConst],1);
        end
        
        powTr = bsxfun(@rdivide, power(freqIdx,:,sets.train), newConst);
        powTr1 = permute(powTr, [1,3,2,4]);
        powTr2 = permute(powTr, [1,3,4,2]);
        cohDenom = bsxfun(@times, powTr1, powTr2);
        cohTr = sqrt(coherence(freqIdx,sets.train,:,:) .* cohDenom);
        
        if holdout
            powVal = bsxfun(@rdivide, power(freqIdx,:,sets.val), newConst);
            powVal1 = permute(powVal, [1,3,2,4]);
            powVal2 = permute(powVal, [1,3,4,2]);
            cohDenom = bsxfun(@times, powVal1, powVal2);
            cohVal = sqrt(coherence(freqIdx,sets.val,:,:) .* cohDenom);
        end
    end
    
    %% calculate power/coherence differences
    powDiff = zeros(F,C,W);
    nPairs = (C.^2 - C)/2;
    cohDiff = zeros(F, nPairs, W);
    if holdout
        powDiff2 = zeros(F,C,W2);
        cohDiff2 = zeros(F, nPairs, W2);
    end
    thisPairIdx = 0;
    for c = 1:C
        powDiff(:,c,:) = squeeze(powTr(:,c,:)) - squeeze(ampUKU(c,c,:,:));
        if holdout
            powDiff2(:,c,:) = squeeze(powVal(:,c,:)) - squeeze(ampUKU2(c,c,:,:));
        else
            powDiff2 = nan;
        end
        for c2 = (c+1):C
            thisPairIdx = thisPairIdx + 1;
            cohDiff(:,thisPairIdx,:) = ...
                squeeze(cohTr(:,:,c,c2)) - squeeze(ampUKU(c,c2,:,:));
            if holdout
                cohDiff2(:,thisPairIdx,:) = ...
                    squeeze(cohVal(:,:,c,c2)) - squeeze(ampUKU2(c,c2,:,:));
            else
                cohDiff2 = nan;
            end
        end
    end
    
    csfaModels{m}.pMseTr = mean(powDiff.^2, 3);
    csfaModels{m}.cMseTr = mean(cohDiff.^2, 3);
    csfaModels{m}.pMseVal = mean(powDiff2.^2, 3);
    csfaModels{m}.cMseVal = mean(cohDiff2.^2, 3);
end

save(modelfile, 'sets', 'csfaModels','labels')
