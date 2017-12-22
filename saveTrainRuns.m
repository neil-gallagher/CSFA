function saveTrainRuns(saveFile, datafile)
% saves results of training runs from checkpoint files into a single file
% with all relevant data
% INPUTS
%   saveFile: path to file where models will be saved
%   dataFile: path to file where relevant data is saved

% add .mat extension if not there
if ~any(regexp(saveFile,'.mat$'))
  saveFile = [saveFile '.mat'];
end

% get a list of all checkpoint files corresponding to saveFile
[pathPart,matches] = strsplit(saveFile,{'\\','/'});
prefix = [];
for i = 1:numel(pathPart)-1
  prefix = [prefix pathPart{i} matches{i}];
end

cpFiles = dir([prefix 'chkpt_*' pathPart{end}]);
N = numel(cpFiles);

% load saved data and initialize list of trained models
if exist(saveFile,'file')
  load(saveFile)
end
if ~exist('csfaModels','var')
  csfaModels = {};
end
  
% add training data from each checkpoint file to one list
for k = 1:N
  trainInfo = load([prefix cpFiles(k).name]);
  
  % every checkpoint file has a sets variable. They should all be
  % the same, so we consolidate.
  if ~exist('sets','var')
    sets = trainInfo.sets;
  end
  trainInfo = rmfield(trainInfo,'sets');
  
  csfaModels = [csfaModels; trainInfo];
end

load(datafile,'labels')

save(saveFile,'csfaModels','sets','labels')