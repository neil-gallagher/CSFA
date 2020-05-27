function [selection, sortingIndices] = selectDiscFactors(numFactors, target, scores, superMask)
% Select Discriminatory Factors
% This takes the CSFA generative model created in
% trainCSFA and finds the factors with the most
% signifciant different between two conditions. Returns indexes of
% most significant factors in a logical array. Significance
% determined using Wilcoxin rank sum test.
% Example:
% dIdx =
% selectDiscFactors(3,labels.windows.genotype(sets.train),scores(sets.train));
%
% Thanks to Kat Hefter for writing the first draft of this function!

L = size(scores,1);

T = numel(target);

% get weighted average log p-value over all classifiers and class values
classVotes = zeros(L,T);
for t = 1:T
    if isa(target{t}, 'cell')
        thisTarget = categorical(target{t});
    else
        thisTarget = target{t};
    end
    supervisedWindows = superMask(:,t);
    
    % get unique identifiers for all classes. consider binary case as one class
    uniqueClasses = unique(thisTarget);
    C = numel(uniqueClasses);
    if C == 2
        uniqueClasses = uniqueClasses(2);
        C = 1;
    end
    
    % for each class, rank factors based on t-test p values
    p = zeros(L,C);
    for c = 1:C
        thisClassLabel = thisTarget == uniqueClasses(c);
        
        for s = 1:L
            posScores = scores(s, thisClassLabel & supervisedWindows);
            negScores = scores(s, (~thisClassLabel) & supervisedWindows);
            [~, thisP] = ttest2(posScores, negScores);
            p(s, c) = thisP;
        end
    end
    
    % break ties by vote 'importance' for this classifier and by classifier
    % order
    avgLogP = mean(log(p),2,'omitnan');
    [~,sortingIndices] = sort(avgLogP,'ascend');
    indx = sortingIndices(1:numFactors);
    classVotes(indx,t) = 1 + 1e-3*(1./(1:numFactors)) + 1e-6*(1./t);
end

[~,sortingIndices] = sort(sum(classVotes,2), 'descend');
indx = sortingIndices(1:numFactors);
selection = false(L,1);
selection(indx) = true;
end
