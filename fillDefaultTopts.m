function trainOpts = fillDefaultTopts(trainOpts)
% fills default trainOptsValues
%   trainOpts : structure of options for the learning
%       algorithm
%       FIELDS
%       iters: total number of iteration. Default: 1000
%       evalInterval(2): interval at which to evaluate likelihood and save a
%           checkpoint. evalInterval2 corresponds to score projection
%           following initial kernel learning. If evalInterval2 is not given,
%           defaults to the value set for evalInterval. evalInterval
%           Default: 20
%       saveInterval: interval at which to save intermediate models during
%           training. Default: 100
%       convThresh(2), convClock(2): training stops if the objective function
%           does not increase by 'convThresh' after 'convClock'
%           evaluations of the log likelihood. convThresh2 and
%           convClock2 correspond to score projection following kernel
%           learning. If values are not set, convThresh2 and convClock2 default
%           to the values set for convThresh and convClock, respectively.
%           convThresh Default: 100. convClock Default: 5.
%       algorithm: function handle to the desired gradient descent
%           algorithm for model learning. default: algorithms.adam
%           Example: [evals,trainModels] = trainOpts.algorithm(labels.s,...
%                          xFft(:,:,sets.train),model,trainOpts,chkptFile);
%       stochastic: boolean indicating to train using mini-batches.
%           Default: true
%       batchSize: (only used if stochastic = true) mini-batch size
%           for stochastic algorithms. Default: 128.
%       projAll: boolean. If false, only learns scores from the final model
%           (obtained after all training iterations), rather than for each
%           intermediate model as well. Default: false
%       normalizeData: boolean. If true, normalizes input (xFft) data so
%           that signal within a given channel/frequency has unit power
%           over all windows (i.e. divide by RMS).
if ~isfield(trainOpts,'iters')
    trainOpts.iters = 1000;
end
if ~isfield(trainOpts,'saveInterval')
    trainOpts.saveInterval = 100;
end
if ~isfield(trainOpts,'evalInterval')
    trainOpts.evalInterval = 20;
end
if ~isfield(trainOpts,'evalInterval2')
    trainOpts.evalInterval2 = trainOpts.evalInterval;
end
if ~isfield(trainOpts,'convThresh')
    trainOpts.convThresh = 100;
end
if ~isfield(trainOpts,'convThresh2')
    trainOpts.convThresh2 = trainOpts.convThresh;
end
if ~isfield(trainOpts,'convClock')
    trainOpts.convClock = 5;
end
if ~isfield(trainOpts,'convClock2')
    trainOpts.convClock2 = trainOpts.convClock;
end
if ~isfield(trainOpts,'algorithm')
    trainOpts.algorithm = @algorithms.rprop;
end
if ~isfield(trainOpts,'stochastic')
    trainOpts.stochastic = true;
end
if ~isfield(trainOpts,'batchSize')
    trainOpts.batchSize = 128;
end
if ~isfield(trainOpts,'fStochastic')
    trainOpts.fStochastic = true;
end
if ~isfield(trainOpts,'fBatchSize')
    trainOpts.fBatchSize = 8;
end
if ~isfield(trainOpts,'projectAll')
    trainOpts.projectAll = false;
end
if ~isfield(trainOpts,'normalizeData')
    trainOpts.normalizeData = true;
end
end