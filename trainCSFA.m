function trainCSFA(loadFile,saveFile,modelOpts,trainOpts,chkptFile)
% trainCSFA
%   Trains a cross-spectral factor analysis (CSFA) model of the given LFP data.
%   Generally, the data are given as averaged signal over each recording area,
%   divided into time windows. Learns a set of factors that describe the dataset
%   well, and models each window as a linear sum of contributions from each
%   factor. The model can be combined with supervised classifier in order to
%   force a set of the factors to be predictive of desired sid information. Run
%   the saveTrainRuns function after this to consolidate training results into
%   one file. For more details on the model refer to the following publication:
%   N. Gallagher, K.R. Ulrich, A. Talbot, K. Dzirasa, L. Carin, and D.E. Carlson,
%     "Cross-Spectral Factor Analysis", Advances in Neural Information
%     Processing Systems 30, pp. 6845-6855, 2017.
%   INPUTS
%   loadFile: path to '.mat' file containing preprocessed data. Must contain
%        variables named xFft and labels variables, described below.
%   saveFile: path to '.mat' file to which the CSFA model will
%       ultimately be saved. If you wish to control the division of
%       data into train/validation/test sets, this file should be
%       already initialized with a sets variable, described below.
%       All models saved to this file should have the same validation sets.
%   modelOpts: (optional) Indicates  parameters of the CSFA model. All
%       non-optional fields not included in the structure passed in will be
%       filled with a default value.
%       FIELDS
%       discrimModel: string indicating the supervised classifier, if any,
%           that is combined with the CSFA model. options are
%           'none','svm','logistic', or 'multinomial'. Default: 'none'
%       L: number of factors. Default: 10
%       Q: number of spectral gaussian components per factor. Default: 3
%       R: rank of coregionalization matrix. Default: 2
%       eta: precision of additive gaussian noise. Default: 5
%       lowFreq: lower bound on spectral frequencies incorporated in the model.
%           Default: 1
%       highFreq: upper bound on spectral frequencies incorporated in the model.
%           Default: 50
%       description: (optional) string description of model
%       kernel: (optional) CSFA model object to initialize new model for
%           training
%       (The following are used if discrimModel is set to anything
%       other than 'none')
%       lambda: scalar ratio of the 'weight' on the classifier loss
%           objective compared to the CSFA model likelihood objective
%       target: string indicating the field of labels.windows to be used as
%           the target variable to be explained by the classifier
%       dIdx: boolean vector of indicies for discriminitive
%           factors
%       mixed: (optional) boolean indicating whether to have mixed intercept
%           model for multiple groups
%       group: Only needed if mixed is set to 'true'. String indicating the
%           field of labels.windows to be used as the group variable for a
%           mixed intercept model
%   trainOpts: (optional) structure of options for the learning algorithm. All
%       non-optional fields not included in the structure passed in will be
%       filled with a default value. See the fillDefaultTopts function for
%       default values.
%       FIELDS
%       iters: maximum number of training iterations
%       evalInterval(2): interval at which to evaluate objective. evalInterval2
%           controls the interval for score learning
%           following initial kernel learning.
%       saveInterval: interval at which to save intermediate models during
%           training.
%       convThresh(2), convClock(2): convergence criterion parameters. training
%           stops if the objective function does not
%           increase by a value of at least (convThresh) after (convClock)
%           evaluations of the objective function. convThresh2 and
%           convClock2 correspond to score learning following kernel
%           learning.
%       algorithm: function handle to the desired gradient descent
%           algorithm for model learning. Stored in +algorithms/
%           Example: [evals,trainModels] = trainOpts.algorithm(labels.s,...
%                          xFft(:,:,sets.train),model,trainOpts,chkptFile);
%       stochastic: boolean indicating to train using mini-batches
%       batchSize: (only used if stochastic = true) mini-batch size
%           for stochastic algorithms
%       projAll: boolean. If false, only learns scores from the final model
%           (obtained after all training iterations), rather than for each
%           intermediate model as well.
%   chkptFile: (optional) path of a file containing checkpoint information
%       for training to start from. For use if training had to be terminated
%       before completion.
%   LOADED VARIABLES
%   (from loadFile)
%   xFft: fourier transform of preprocessed data. NxAxW array. A is
%       the # of areas. N=number of frequency points per
%       window. W=number of time windows.
%   labels: Structure containing labeling infomation for data
%       FIELDS
%       s: frequency space (Hz) labels of fourier transformed data
%   (optionally loaded from saveFile)
%   sets: structure containing
%       train/validation set labels.
%       FIELDS
%       train: logical vector indicating windows in xFft to be used
%           in training set
%       val: (optional) logical vector indicating window to be used in
%           validation
%       datafile: path to file containing data used to train model
%       test: (optional) logical vector indicating windows for
%           testing
%       description: (optional) describes validation set scheme
%
% Example1: TrainCSFA('data/dataStore.mat','data/modelFile.mat',mOpts,tOpts)
% Example2: TrainCSFA('data/dataStore.mat','data/Mhold.mat',[],[],'data/chkpt_81LNf_Mhold.mat')

% Add .mat extension if filenames don't already include them
saveFile = addExt(saveFile);
loadFile = addExt(loadFile);

% load data and associated info
load(loadFile,'xFft','labels')
nWin = size(xFft,3);
sets = loadSets(saveFile,loadFile,nWin);
holdout = ~sets.train;

% initialize options structures if not given as inputs
if nargin < 4
    trainOpts = [];
end
if nargin < 3
    modelOpts = [];
end

if nargin < 5
    % fill in default options and remaining parameters
    modelOpts.C = size(xFft,2); % number of signals
    modelOpts.W = sum(sets.train);    % # of windows
    
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
    if isfield(cp,'scores'), scores = cp.scores; end
    if isfield(cp,'evals'), evals = cp.evals; end
end

modelOpts = fillDefaultMopts(modelOpts);
trainOpts = fillDefaultTopts(trainOpts);

if trainOpts.normalizeData
    % normalize data to have unit power for each channel/frequency
    normConst = sqrt(mean(abs(xFft).^2,3));
    xFft = bsxfun(@rdivide, xFft, normConst);
    save(chkptFile,'normConst','-append')
end

%% Kernel learning
% train kernels if they haven't been loaded from chkpt file
if ~exist('scores','var') && (~exist('trainIter','var') || trainIter~=Inf)
    
    if exist('trainModels','var') % implies chkptFile was loaded
        model = trainModels(end);
    else
        model = initModel(modelOpts,labels,sets,xFft);
    end
    
    % update model via gradient descent
    [evals, trainModels] = trainOpts.algorithm(labels.s,xFft(:,:,sets.train),model,...
        trainOpts,chkptFile);
    fprintf('Kernel Training Complete\n')
end

%% Score learning
% Fix kernels and learn scores to convergence

% initialize variables for projection
nModels = numel(trainModels);
if exist('scores','var')
    % determine idx of last projected model if there are saved score projection
    % models. Initialize learned scores with scores from previous model.
    [~,~,k] = ind2sub([modelOpts.L,nWin,nModels],find(scores,1));
    initScores = scores(:,holdout,k);
    k=k-1;
else
    k = nModels;
    scores = zeros(modelOpts.L,nWin,nModels);
    
    nHoldout = sum(holdout);
    initScores = nan(modelOpts.L,nHoldout);
end

while k >= lastTrainIdx(nModels,trainOpts.projectAll)
    % for each saved model learn scores for training & holdout sets
    if isa(trainModels(k),'GP.CSFA')
        thisTrainModel = trainModels(k);
    else
        thisTrainModel = trainModels(k).kernel;
    end
    
    a = tic;
    thisTrainModel = projectCSFA(xFft(:,:,sets.train),thisTrainModel,labels.s,trainOpts,...
        thisTrainModel.scores);
    scores(:,sets.train,k) = thisTrainModel.scores;

    % save updated training model
    if isa(trainModels(k),'GP.CSFA')
        trainModels(k) = thisTrainModel;
    else
        trainModels(k).kernel = thisTrainModel;
    end

    save(chkptFile,'scores','trainModels','evals',...
        'modelOpts','trainOpts','sets')
    
    if sum(holdout) > 0
        holdoutModels(k) = projectCSFA(xFft(:,:,holdout),thisTrainModel,labels.s,...
                                       trainOpts, initScores);
        scores(:,holdout,k) = holdoutModels(k).scores;

        % initialize next model with scores from current model
        initScores = holdoutModels(k).scores;
        
        save(chkptFile,'holdoutModels','-append')
    end

    fprintf('Projected model %d: %.1fs\n',k,toc(a))
    
    k = k-1;
end
if ~trainOpts.projectAll
    scores = scores(:,:,end);
    save(chkptFile,'scores','-append')
end
end


function sets = loadSets(saveFile,loadFile,nWin)
% load validation set options

if exist(saveFile,'file')
    load(saveFile,'sets')
    % allow user to set sets.train to true for training set
    % to include all data
    if sets.train == true
        sets.train = true(1,nWin);
    end
else
    sets.train = false(1,nWin);
    sets.train(randperm(nWin,floor(0.8*nWin))) = 1;
    sets.test = ~sets.train;
    sets.description = '80/20 split';
    sets.datafile = loadFile;
end
end

function labelArr = compileLabels(id, labels, trainIdx)
id = toCell(id);

K = length(id);
labelArr = cell(1,K);
for k = 1:K
    if strcmp(id{k}, 'all')
        labelArr{k} = ones(sum(trainIdx),1);
    else
        labelArr{k} = labels.windows.(id{k})(trainIdx);
    
        %make sure labels are column vectors
        if size(labelArr{k},2) > size(labelArr{k},1)
            labelArr{k} = labelArr{k}';
        end
    end
end
end


function model = initModel(modelOpts, labels, sets, xFft)
% initialize CSFA or dCSFA model

% if supervised, set up target labels and supervision window mask
modelOpts.discrimModel = toCell(modelOpts.discrimModel);
if ~strcmp(modelOpts.discrimModel{1},'none')
    targetLabel = compileLabels(modelOpts.target, labels, sets.train);
    
    iwsCell = compileLabels(modelOpts.supervisedWindows,labels,sets.train);
    modelOpts.isWindowSupervised = cell2mat(cellfun(@(x) logical(x), iwsCell,...
        'UniformOutput',false));
    
    if isa(modelOpts.balance, 'cell') || any(modelOpts.balance)
        if ~isa(modelOpts.balance, 'cell')
           modelOpts.balance = {modelOpts.balance}; 
        end
        T = numel(targetLabel);
        modelOpts.classWeights = cell(T,1);
        
        for t = 1:T
            % get mouse, target, and group labels to be used by each classifier
            thisIdx = modelOpts.isWindowSupervised(:,t);
            thisM = labels.windows.mouse(sets.train); thisM = thisM(thisIdx);
            thisT = targetLabel{t}(thisIdx);
            
            % generate list of groups to recursively balance data over
            balanceGroups = {};
            for b = 1:numel(modelOpts.balance)
                thisG = labels.windows.(modelOpts.balance{b})(sets.train);
                thisG = thisG(thisIdx);
                balanceGroups = [balanceGroups, {thisG}];
            end
            balanceGroups = [{thisT}, balanceGroups, {thisM}];
            
            if numel(unique(thisT)) > 2
                warning('Multinomial classifier will not handle observations weights for balancing')
                continue
            end
            
            % calculate window weightings
            modelOpts.classWeights{t} = util.balancedWeights(balanceGroups);
        end
    end
end

if isa(modelOpts.discrimModel{1},'function_handle')
    model = GP.dCSFA(modelOpts,targetLabel);
else
    xFftTrain = xFft(:,:,sets.train);
    switch modelOpts.discrimModel{1}
        case 'none'
            model = GP.CSFA(modelOpts, labels.s, xFftTrain);
        case {'svm', 'logistic', 'multinomial'}
            if isfield(modelOpts, 'group')               
                group = compileLabels(modelOpts.group, labels, sets.train);
                model = GP.dCSFA(modelOpts, targetLabel, group, labels.s, xFftTrain);
            else
                model = GP.dCSFA(modelOpts, targetLabel, [], labels.s, xFftTrain);
            end
        otherwise
            warning(['Disciminitive model indicated by modelOpts.discrimModel is '...
                'not valid. Model will be trained using GP generative model only.'])
            model = GP.CSFA(modelOpts,labels.s,xFftTrain);
    end
end
end

function chkptFile = generateCpFilename(saveFile)
% generate checkpoint file name that wont overlap with other checkpoint
% files for same dataset

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

function modelOpts = fillDefaultMopts(modelOpts)
% fill in default model options
if ~isfield(modelOpts,'L'), modelOpts.L = 10; end
if ~isfield(modelOpts,'Q'), modelOpts.Q = 3; end
if ~isfield(modelOpts,'R'), modelOpts.R = 2; end
if ~isfield(modelOpts,'eta'), modelOpts.eta = 5; end
if ~isfield(modelOpts,'lowFreq'), modelOpts.lowFreq = 1; end
if ~isfield(modelOpts,'highFreq'), modelOpts.highFreq = 50; end
if ~isfield(modelOpts,'maxW')
    modelOpts.maxW = min(modelOpts.W,1e4);
end
if ~isfield(modelOpts,'learnNoise'), modelOpts.learnNoise = true; end
if ~isfield(modelOpts,'regB'), modelOpts.regB = 100; end
if ~isfield(modelOpts,'discrimModel'), modelOpts.discrimModel = 'none'; end

if ~strcmp(modelOpts.discrimModel, 'none')
% set default dCSFA parameters
   if ~isfield(modelOpts,'lambda'), modelOpts.lambda = 1e3; end
   if ~isfield(modelOpts,'dIdx'), modelOpts.dIdx = 1; end
   if ~isfield(modelOpts,'supervisedWindows'), modelOpts.supervisedWindows = 'all'; end
   if ~isfield(modelOpts,'balance'), modelOpts.balance = false; end
end
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

function obj = toCell(obj)
if ~isa(obj, 'cell')
    obj = {obj};
end
end

