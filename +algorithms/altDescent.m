function [evals,trainModels] = altDescent(x,yAll,model,opts,algVars,chkptfile)
% descent : general gradient descent (ascent)
% Inputs
%    x - function inputs (e.g. frequencies)
%    yAll - function outputs (e.g. frequency domain signal)
%    model - model of the function between x and y; must contain methods
%            'gradient', 'evaluate', 'getParams', 'setParams',
%            and 'getBounds'
%    opts - options for training chosen by user
%       FIELDS
%       iters: total number of iteration
%       evalInterval: interval at which to evaluate likelihood and save a
%           checkpoint.
%       saveInterval: interval at which to save intermediate models during
%           training.
%       convThresh, convClock: training stops if the objective function
%           does not increase by 'convThresh' after 'convClock'
%           evaluations of the log likelihood.
%       batchSize: mini-batch size for stochastic algorithms
%       stochastic: boolean indicating stochastic algorithm is being used
%    algVars - structure of variables for specific learning
%        algorithm to be used
%        FIELDS
%            calcStep: function handle for the update rule of the
%            desired learning algorithm. Takes a gradient vector
%            and algVars as inputs and returns the update vector
%            and updated algVars.
%    chkptfile (optional) - path to file where checkpoint models will be
%        saved
%
% Output
%    evals - function evaluations every 'evalInterval' iterations
%    trainModels - intermediate models saved during training

% initialize things
iter = 1;
convCntr = 0;
maxEval = -Inf;
nModelSaves = ceil(opts.iters/opts.saveInterval);
trainModels(nModelSaves) = model;
saveIdx = 1;

evals = zeros(1,opts.iters);
rEvals = zeros(1,opts.iters);
if isa(model, 'GP.dCSFA')
    sEvals = zeros(opts.iters, model.S);
end
condNum = zeros(1,opts.iters);
W = size(yAll,3);
Ns = sum(model.freqBand(x));
tic

% load info in checkpoint file (Check that this works)
if nargin >= 6
    cp = load(chkptfile);
    if all(isfield(cp,{'params', 'algVars', 'trainIter'}))
        algVars = cp.algVars;
        model.setParams(cp.params);
        iter = cp.trainIter;
        if isfield(cp,'trainModels'), trainModels = cp.trainModels; end
        if isfield(cp,'evals'), evals = cp.evals; end
        if isfield(cp,'rEvals'), rEvals = cp.rEvals; end
        if isa(model, 'GP.dCSFA')
            if isfield(cp,'sEvals'), sEvals = cp.sEvals; end
        end
    end
else
    model.makeIdentifiable();
end

% remember not to update kernels if set to false
updateKernels = model.updateKernels;
paramIdx = model.getParamIdx;
while (iter <= opts.iters) && (convCntr < opts.convClock)
    % get gradients for and update scores first
    model.setUpdateState(false, true);
    if opts.fStochastic
        fInds = false(1,Ns);
        fInds(randsample(Ns, opts.fBatchSize)) = true;
        [g, condNum(iter)] = model.gradient(x,yAll,[],fInds);
    else
        [g, condNum(iter)] = model.gradient(x,yAll);
    end
    params = model.getParams();
    updateIdx = paramIdx.scores | paramIdx.noise;
    [step, algVars] = algVars.calcStep(g, algVars, updateIdx);
    pNew = params + step;
    model.setParams(pNew);
    
    if condNum(iter) > 1e6
        warning(['Condition number on covariance matrix is %.3e. Gradient ' ...
            'calculation are likely incorrect.'],condNum(iter))
    end

    if updateKernels
        % get gradients for and update kernel parameters
        model.setUpdateState(true,false)

        if opts.stochastic
            % use minibatch of windows to calculate gradient
            inds = randsample(W,opts.batchSize);
            y = yAll(:,:,inds);
            if opts.fStochastic
                [g, ~] = model.gradient(x,y,inds,fInds);
            else
                [g, ~] = model.gradient(x,y,inds);
            end
        else
            if opts.fStochastic
                [g, ~] = model.gradient(x,yAll,[],fInds);
            else
                [g, ~] = model.gradient(x,yAll);
            end
        end
        
        params = model.getParams();
        updateIdx = ~paramIdx.scores;
        [step, algVars] = algVars.calcStep(g, algVars, updateIdx);
        pNew = params + step;
        model.setParams(pNew);
    end
    model.setUpdateState(updateKernels,true);
    
    % occasionally check performance
    if mod(iter,opts.evalInterval)==0
        if isa(model, 'GP.dCSFA')
            [llEval, rLoss, cLoss] = model.evaluate(x,yAll);
            sEvals(iter,:) = cLoss;
            totalEval = llEval - cLoss*model.lambda' - rLoss*model.kernel.regB;
        else
            [llEval, rLoss] = model.evaluate(x,yAll);
            totalEval = llEval - rLoss*model.regB;
        end
        evals(iter) = llEval;
        rEvals(iter) = rLoss;
            
        if opts.stochastic
            winsComplete = iter*opts.batchSize;
            fprintf(['Iter #%5d/%d - %.1f Epochs Completed - Time:%4.1fs - LL:%4.4g - '...
                'Max Cond. #: %.3g'], iter, opts.iters, winsComplete/W, ...
                toc, llEval, condNum(iter))
        else
            fprintf('Iter #%5d/%d - Time:%4.1fs - LL:%4.4g - Max Cond. #: %.3g',...
                iter, opts.iters, toc, llEval, condNum(iter))
        end
        if isa(model, 'GP.dCSFA')
            fprintf(' - Sup. Loss:%4.4g', cLoss)
        end
        if updateKernels
           fprintf(' - Reg. Loss:%4.4g', rLoss) 
        end
        fprintf('\n')
        
        if nargin >= 6
            cp.evals = evals;
            cp.rEvals = rEvals;
            if isa(model, 'GP.dCSFA')
                cp.sEvals = sEvals;
            end
        end
        
        % convergence check
        if totalEval > maxEval + opts.convThresh
            convCntr = 0;
            maxEval = totalEval;
        else
            convCntr = convCntr + 1;
        end
    end
    
    % occasionally save intermediate models
    if mod(iter,opts.saveInterval)==0
        mCopy = model.copy;
        mCopy.makeIdentifiable();
        trainModels(saveIdx) = mCopy;
        saveIdx = saveIdx + 1;
        fprintf('Saved model after %d iterations\n',iter);
        if nargin >= 6
            cp.trainModels = trainModels;
        end
    end
    
    % save info back to checkpoint file
    if nargin >= 6
        cp.params = model.getParams(); cp.algVars = algVars;
        cp.trainIter = iter+1;
        save(chkptfile,'-struct','cp')
    end
    
    iter = iter + 1;
end
iter = iter - 1;

if isfinite(iter)
    % save final results if necessary
    if mod(iter,opts.evalInterval)~=0
        if isa(model, 'GP.dCSFA')
            [llEval, rLoss, cLoss] = model.evaluate(x,yAll);
            sEvals(iter,:) = cLoss;
        else
            [llEval, rLoss] = model.evaluate(x,yAll);
        end
        evals(iter) = llEval;
        rEvals(iter) = rLoss;
        
        if nargin >= 6
            cp.evals = evals;
            cp.rEvals = rEvals;
            if isa(model, 'GP.dCSFA')
                cp.sEvals = sEvals;
            end
        end
    end
    if mod(iter,opts.saveInterval)~=0
        model.makeIdentifiable();
        trainModels(saveIdx) = model;
        saveIdx = saveIdx + 1;
        if nargin >= 6
            cp.trainModels = trainModels;
        end
    end
    if nargin >= 6
        cp.params = model.getParams();
        cp.trainIter = Inf;
        save(chkptfile,'-struct','cp')
    end
end

% remove unused elements in trainModels
trainModels(saveIdx:end) = [];
end