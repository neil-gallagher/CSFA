function trainOpts = fillDefaultTopts(trainOpts)
% fills default trainOptsValues
%   trainOpts : structure of options for the learning
%       algorithm
%       FIELDS
%       iters: total number of iteration
%       evalInterval(2): interval at which to evaluate likelihood and save a
%           checkpoint. evalInterval2 corresponds to score projection
%           following initial kernel learning.
%       saveInterval: interval at which to save intermediate models during
%           training. 
%       convThresh(2), convClock(2): training stops if the objective function
%           does not increase by 'convThresh' after 'convClock'
%           evaluations of the log likelihood. convThresh2 and
%           convClock2 correspond to score projection following kernel
%           learning.
%       algorithm: function handle to the desired gradient descent
%           algorithm for model learning. 
%           Example: [evals,trainModels] = trainOpts.algorithm(labels.s,...
%                          xFft(:,:,sets.train),model,trainOpts,chkptFile);
%       stochastic: boolean indicating to train using mini-batches
%       batchSize: (only used if stochastic = true) mini-batch size
%           for stochastic algorithms 
%       projFinal: boolean. only project scores from the final model
%           (obtained after all training iterations)
  if ~isfield(trainOpts,'iters'), trainOpts.iters = 5e4; end
  if ~isfield(trainOpts,'saveInterval'), trainOpts.saveInterval = 50; end 
  if ~isfield(trainOpts,'evalInterval'), trainOpts.evalInterval = 10; end
  if ~isfield(trainOpts,'evalInterval2'), trainOpts.evalInterval2 = 5; end
  if ~isfield(trainOpts,'convThresh'), trainOpts.convThresh = 100; end
  if ~isfield(trainOpts,'convThresh2'), trainOpts.convThresh2 = 10; end
  if ~isfield(trainOpts,'convClock'), trainOpts.convClock = 5; end
  if ~isfield(trainOpts,'convClock2'), trainOpts.convClock2 = 5; end
  if ~isfield(trainOpts,'algorithm')
    trainOpts.algorithm = @algorithms.adam;
  end
  if ~isfield(trainOpts,'stochastic'), trainOpts.stochastic = false; end
  if ~isfield(trainOpts,'batchSize'), trainOpts.batchSize = 500; end
  if ~isfield(trainOpts,'projectAll'), trainOpts.projectAll = false; end
end