function [evals,trainModels] = gd(x,y,model,opts,chkptfile)
% gd: basic gradient descent (ascent)
% Inputs
%    x - function inputs
%    y - function outputs
%    model - model of the function between x and y; must contain methods
%            'gradient', 'evaluate', 'getParams', 'setParams',
%            and 'getBounds'
%    opts - options, including 'iters', 'evalInterval', and any model opts
%       FIELDS
%       iters: total number of iteration
%       evalInterval: interval at which to evaluate likelihood and save a
%           checkpoint
%       saveInterval: interval at which to save intermediate models during
%           training
%       convThresh: convergence threshold (average change in parameters)
%    chkptfile (optional) - path to file where checkpoint models will be
%        savedtrainMode
%
% Output
%    evals - function evaluations every 'evalInterval' iterations
%    trainModels - intermediate models saved during training
LEARN_RATE = 1e-6;

algVars.calcStep = @(g,av) calcStep(g,av,LEARN_RATE);

if nargin > 4
  [evals,trainModels] = algorithms.descent(x,y,model,opts,algVars,chkptfile);
else
  [evals,trainModels] = algorithms.descent(x,y,model,opts,algVars);
end

end

function [step,a] = calcStep(g,a,learnRate)
step = g*learnRate;
end