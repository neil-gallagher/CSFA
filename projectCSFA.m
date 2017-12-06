function modelRefit = projectCSFA(xFft,origModel,dataOpts,trainOpts,initScores)
% projectCSFA
%   Projects a dataset onto the space of factors for a
%   cross-spectral factor analysis (CSFA) model
%   INPUTS
%   xFft: fourier transform of preprocessed data. NxAxW array. A is
%     the # of areas. N=number of frequency points per
%     window. W=number of time windows.
%   origModel: CSFA model containing factors onto which you desire
%     to project a new dataset (xFft)
%   dataOpts: struct containing parameters describing the data
%     FIELDS
%     highFreq/lowFreq: boundaries of frequencies considered by
%       the model
%     s: vector of frequencies corresponding to the first dimension
%       of xFft
%   trainOpts: (optional) structure containing parameters for the
%     training algorithm
%       FIELDS
%       iters: total number of iteration
%       evalInterval: interval at which to evaluate likelihood and save a
%           checkpoint.
%       convThresh, convClock: training stops if the objective function
%           does not increase by 'convThresh' after 'convClock'
%           evaluations of the log likelihood.
%       algorithm: function handle to the desired gradient descent
%           algorithm for model learning. 
%           Example: [evals,trainModels] = trainOpts.algorithm(labels.s,...
%                          xFft(:,:,sets.train),model,trainOpts,chkptFile);
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
trainOpts.evalInterval = trainOpts.evalInterval2;
trainOpts.convThresh = trainOpts.convThresh2;
trainOpts.convClock = trainOpts.convClock2;

% model parameters
modelRefitOpts = extractModelOpts(origModel);
modelRefitOpts.W = size(xFft,3);
modelRefitOpts.maxW = min(1e3,modelRefitOpts.W);

% initialize new model
kernels = origModel.LMCkernels;
modelRefit = GP.CSFA(modelRefitOpts,dataOpts,kernels);
modelRefit.updateKernels = false;
if nargin >= 5
  scoresGiven = ~isnan(initScores);
  modelRefit.scores(scoresGiven) = initScores(scoresGiven);
end

% project new data onto factors
trainOpts.algorithm(dataOpts.s,xFft,modelRefit,trainOpts);
end

function modelOpts = extractModelOpts(model)
  modelOpts.L = model.L;
  modelOpts.Q = model.Q;
  modelOpts.C = model.C;
  modelOpts.R = model.LMCkernels{1}.coregs.B{1}.R;
  modelOpts.eta = model.eta;
end