function trainCSFA(loadFile,saveFile,modelOpts,trainOpts,chkptFile)
% trainCSFA
%   Trains a cross-spectral factor analysis (CSFA) model to model
%   LFP data averaged over each recording area. The
%   model can be combined with a discriminitive supervised model. Must
%   run saveTrainRuns afterward to consolidate training results into file
%   with data
%   INPUTS
%   loadFile: path of file containing preprocessed data. Should contain
%       xFft, dataOpts and labels variables, described below.
%   saveFile: name of '.mat' file to which the CSFA model is
%       ultimately saved. If you wish to control the division of
%       data into train/validation/test sets, this file should be
%       already initialized with a sets variable, described below.
%       All models saved to this file should have the same sets.
%   modelOpts (optional): Indicates  parameters of the CSFA model.
%       FIELDS
%       discrimModel: string indicating the discriminitve model, if any,
%           that is combined with the GP model. options are
%           'none','svm', or 'logistic'; this can also be a
%           function handle to a custom classifier ?!?
%       L: number of factors
%       Q: number of spectral gaussians components per factor
%       R: rank of coregionalization matrix
%       eta: precision of additive gaussian noise
%       description (optional): description of model
%       kernel (optional): CSFA model to initialize model for
%           training with
%       (The following are used if discrimModel is set to anything
%       other than 'none')
%       dIdx: boolean vector of indicies for discriminitive
%           factors
%       lambda: scalar ratio of the 'weight' on the discriminitive
%           objective compared to the generative likelihood
%           objective
%       target: string indicating the field of labels.windows to be used as
%           the target variable to be explained by the discriminitive model 
%       mixed(optional): boolean indicating whether to have mixed intercept
%           model for multiple groups
%   trainOpts (optional): structure of options for the learning
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
%   chkptFile (optional): path of a file containing checkpoint information
%       for training to start from
%   LOADED VARIABLES
%   dataOpts: Data preprocessing options.
%       FIELDS
%       highFreq/lowFreq: boundaries of frequencies considered by
%       the model
%       subSampFact: subsampling factor
%       normWindows: boolean indication whether to normalize
%           individual windows. If false, dataset is normalized as
%           a whole, and individual windows are still mean
%           subtracted.
%   xFft: fourier transform of preprocessed data. NxAxW array. A is
%       the # of areas. N=number of frequency points per
%       window. W=number of time windows.
%   labels: Structure containing labeling infomation for data
%       FIELDS
%       s: frequency space labels of fourier transformed data
%       target: boolean vector giving binary class labels for
%           discriminitive models
%       group: vector giving group labels for mixed intercept model (see
%           modelOpts entry)
%   sets (optionally loaded from loadFile): structure containing
%       train/validation set labels.
%       FIELDS
%       train: logical vector indicating windows in xFft to be used
%           in training set
%       val(optional): logical vector indicating window to be used in
%           validation
%       datafile: path to file containing data used to train model
%       test (optional): logical vector indicating windows for
%           testing
%       description (optional): describes validation set scheme
% Example: TrainCSFA('data/dataStore.mat','data/Mhold.mat',[],[],'data/chkpt_81LNf_Mhold.mat')

  saveFile = addExt(saveFile);
  loadFile = addExt(loadFile);
  
  if nargin < 4
    trainOpts = [];
  end
  if nargin < 3
    modelOpts = [];
  end
  
  load(loadFile,'xFft','dataOpts','labels')
  nWin = size(xFft,3);

  % validation set options
  if exist(saveFile,'file')
    load(saveFile,'sets')
    % allow user to set sets.train to true for training set
    % to include all data
    if sets.train == true
      sets.train = true(1,nWin);
    end
  else
    sets = randomSplit(nWin,loadFile);
  end
  
  if nargin < 5
    % initialize matfile for checkpointing
    chkptFile = generateCpFilename(saveFile)
    save(chkptFile,'modelOpts','trainOpts','sets')
  else
    % load info from checkpoint file
    chkptFile = addExt(chkptFile)
    cp = load(chkptFile,'-mat');
    modelOpts = cp.modelOpts;
    trainOpts = cp.trainOpts;
    if isfield(cp,'trainIter'), trainIter = cp.trainIter; end
    if isfield(cp,'trainModels'), trainModels = cp.trainModels; end
    if isfield(cp,'projModels'), projModels = cp.projModels; end
    if isfield(cp,'evals'), evals = cp.evals; end
  end

  % fill in default options and remaining parameters
  trainOpts = fillDefaultTopts(trainOpts);
  modelOpts = fillDefaultMopts(modelOpts);
  modelOpts.C = size(xFft,2); % number of signals
  modelOpts.W = sum(sets.train);    % # of windows
  if ~isfield(modelOpts,'maxW')
    modelOpts.maxW = min(modelOpts.W,1e4);
  end
  if ~isfield(modelOpts,'discrimModel')
    modelOpts.discrimModel = 'none';
  end

  % train kernels if they haven't been loaded from chkpt file
  if ~exist('projModels','var') && (~exist('trainIter','var') || trainIter~=Inf)
    
    % Initialize model
    if exist('trainModels','var') % implies chkptFile was loaded
      model = trainModels(end);
    else
      if isa(modelOpts.discrimModel,'function_handle')
        target = modelOpts.target;
        model = GP.dCSFA(modelOpts,dataOpts,labels.windows.(target)(sets.train));
      else
        switch modelOpts.discrimModel
          case 'none'
            model = GP.CSFA(modelOpts,dataOpts);
          case {'svm','logistic','multinomial'}
            target = modelOpts.target;
            if isfield(modelOpts,'mixed') && modelOpts.mixed
              model = GP.dCSFA(modelOpts,dataOpts,labels.windows.(target)(sets.train),...
                               labels.group(sets.train));
            else
              model = GP.dCSFA(modelOpts,dataOpts,labels.windows.(target)(sets.train));
            end
          otherwise
            warning(['Disciminitive model indicated by modelOpts.discrimModel is '...
                     'not valid. Model will be trained using GP generative model only.'])
            model = GP.CSFA(modelOpts,dataOpts);
        end
      end
    end
    
    % update model via resilient backpropagation
    [evals, trainModels] = trainOpts.algorithm(labels.s,xFft(:,:,sets.train),model,...
                                            trainOpts,chkptFile);  
    fprintf('Kernel Training Complete\n')
  end

  % initialize variables for projection
  nModels = numel(trainModels);
  if exist('projModels','var')
    % happens if there are checkpointed test models
    k = nModels - sum(~isempty(projModels));
    initScores = projModels(k+1).scores;
  else
    k = nModels;

    % initialize projected scores with trainin set scores
    initScores = nan(modelOpts.L,nWin);
    initScores(:,sets.train) = trainModels(k).scores;
  end
  
  dataOpts.s = labels.s;
  while k >= lastTrainIdx(nModels,trainOpts.projectAll)
    if isa(trainModels(k),'GP.CSFA')
      thisTrainModel = trainModels(k);
    else
      thisTrainModel = trainModels(k).kernel;
    end
    % update model via resilient backpropagation (requires data.y as input)
    a = tic;
    modelRefit = projectCSFA(xFft,thisTrainModel,dataOpts,trainOpts,...
        initScores);
    fprintf('Trained holdout model %d: %.1fs\n',k,toc(a))
    projModels(k) = modelRefit.copy;

    save(chkptFile,'projModels','trainModels','evals',...
      'modelOpts','trainOpts','sets')

    % initialize next model with scores from current model
    initScores = modelRefit.scores;
    k = k-1;
  end
end

% generate checkpoint file name that wont overlap with other checkpoint
% files for same dataset
function chkptFile = generateCpFilename(saveFile)
  % generate random string
  symbols = ['a':'z' 'A':'Z' '0':'9'];
  ST_LENGTH = 5;
  nums = randi(numel(symbols),[1 ST_LENGTH]);
  st = ['chkpt_' symbols(nums),'_'];

  % add chkpt_ to start of filename (taking path into account)
  idx = regexp(saveFile,'[//\\]');
  if isempty(idx), idx = 0; end
  chkptFile = [saveFile(1:idx(end)),st,saveFile(idx(end)+1:end)];
end

function sets = randomSplit(nWin,loadFile)
    sets.train = false(1,nWin);
    sets.train(randperm(nWin,floor(0.8*nWin))) = 1;
    sets.test = ~sets.train;
    sets.description = '80/20 split';
    sets.datafile = loadFile;
end

function modelOpts = fillDefaultMopts(modelOpts)
  if ~isfield(modelOpts,'L'), modelOpts.L = 21; end
  if ~isfield(modelOpts,'Q'), modelOpts.Q = 3; end
  if ~isfield(modelOpts,'R'), modelOpts.R = 2; end
  if ~isfield(modelOpts,'eta'), modelOpts.eta = 5; end
end

function filename = addExt(filename)
% add .mat extension if not there
  if ~any(regexp(filename,'.mat$'))
    filename = [filename '.mat'];
  end
end

function idx = lastTrainIdx(nModels,projectAll)
  if projectAll
    idx = 1;
  else 
    idx = nModels;
  end
end