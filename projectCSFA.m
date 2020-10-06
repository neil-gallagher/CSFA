function modelRefit = projectCSFA(xFft,origModel,s,trainOpts,initScores)
% projectCSFA
%   Projects a dataset onto the space of factors for a
%   cross-spectral factor analysis (CSFA) model
%   INPUTS
%   xFft: fourier transform of preprocessed data. NxAxW array. A is
%     the # of channels. N=number of frequency points per
%     window. W=number of time windows.
%   origModel: CSFA model containing factors onto which you desire
%     to project the dataset (xFft)
%   s: frequency space (Hz) labels of fourier transformed data
%   trainOpts: (optional) structure of options for the learning algorithm. All
%       non-optional fields not included in the structure passed in will be
%       filled with a default value. See the fillDefaultTopts function for
%       default values.
%     FIELDS
%       iters: maximum number of training iterations. Default: 1000
%       evalInterval2: interval at which to evaluate objective and print
%           feedback during initial training. Default: 20
%       convThresh2, convClock2: convergence criterion parameters. training
%           stops if the objective function does not increase
%           by a value of at least (convThresh2) after (convClock2)
%           evaluations of the objective function.
%           convThresh2 Default: 10; convClock2 Default: 5
%       algorithm: function handle to the desired gradient descent
%           algorithm for model learning. Stored in +algorithms/
%           Default: @algorithms.rprop
%   initScores: (optional) LxW of scores to initialize
%     projection. L = number of factors. W = last dimension of
%     xFft. NaN entries in initScores will be replaced with a
%     random initialization.

if nargin < 4
    trainOpts = [];
end
trainOpts = fillDefaultTopts(trainOpts);

% adjust training options to be appropriate for score projection
if isequal(trainOpts.algorithm,@algorithms.noisyAdam)
    trainOpts.algorithm = @algorithms.adam;
end
trainOpts.saveInterval = trainOpts.iters + 1;
trainOpts.stochastic = false;
trainOpts.fStochastic = false;
trainOpts.evalInterval = trainOpts.evalInterval2;
trainOpts.convThresh = trainOpts.convThresh2;
trainOpts.convClock = trainOpts.convClock2;

% initialize new model
modelRefit = origModel.copy();
modelRefit.updateKernels = false;
modelRefit.updateNoise = false;
modelRefit.regB = 0;
W = size(xFft,3);
maxW = min(origModel.maxW,W);
modelRefit.setPartitions(W, maxW);

% initialize scores
scores = randsample(origModel.scores(:), W*modelRefit.L, true);
modelRefit.scores = reshape(scores, [modelRefit.L, W]);
if nargin >= 5
    scoresGiven = ~isnan(initScores);
    modelRefit.scores(scoresGiven) = initScores(scoresGiven);
end

% project new data onto factors
trainOpts.algorithm(s,xFft,modelRefit,trainOpts);
end

function tOpts = fillDefaultTopts(tOpts)
% fill in default training options
if ~isfield(tOpts,'iters'), tOpts.iters = 1000; end
if ~isfield(tOpts,'evalInterval2'), tOpts.evalInterval = 20; end
if ~isfield(tOpts,'convThresh2'), tOpts.convThresh = 10; end
if ~isfield(tOpts,'convClock2'), tOpts.convClock = 5; end
if ~isfield(tOpts,'algorithm'), tOpts.algorithm = @algorithms.rprop; end
end