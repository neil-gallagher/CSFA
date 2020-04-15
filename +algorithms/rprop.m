function [evals,trainModels] = rprop(x,y,model,opts,chkptfile)
% RPROP   Resilient backpropagation function for gradient ascent
%   implements modified version of iRprop- w/ parameter bounds
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
%        saved
%
% Output
%    evals - function evaluations every 'evalInterval' iterations
%    trainModels - intermediate models saved during training
STEP_MAX = 10;
STEP_MIN = 1e-6;

nParams = numel(model.getParams);
algVars.g_old = zeros(nParams,1);
algVars.step = .01*ones(nParams,1);
algVars.calcStep = @(g,av) calcStep(g,av,STEP_MAX,STEP_MIN);

if nargin > 4
    [evals,trainModels] = algorithms.descent(x,y,model,opts,algVars,chkptfile);
else
    [evals,trainModels] = algorithms.descent(x,y,model,opts,algVars);
end

end

function [step, algVars] = calcStep(g,algVars,stepMax,stepMin)
% adjust step sizes
gg = g.*algVars.g_old;
step = algVars.step;
step(gg>0) = min(1.2*algVars.step(gg>0),stepMax);
step(gg<0) = max(0.5*algVars.step(gg<0),stepMin);
step = step.*sign(g);

algVars.step = step;
algVars.g_old = g;
end