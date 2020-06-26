function saveTrainRuns(saveFile, datafile, cpDir)
% saves results of training runs from checkpoint files into a single file
% with all relevant data
% INPUTS
%   saveFile: path to file where models will be saved
%   datafile: path to file where relevant data is saved
%   cpDir: path to directory where checkpoint files are saved

% add .mat extension if not there
if ~any(regexp(saveFile,'.mat$'))
    saveFile = [saveFile '.mat'];
end

% add '/' to end of cpDir if not there
if ~(cpDir(end)==filesep)
    cpDir = [cpDir filesep];
end

% get a list of all checkpoint files corresponding to saveFile
[pathPart,~] = strsplit(saveFile,{'\\','/'});
cpFiles = dir([cpDir 'chkpt_*' pathPart{end}]);
N = numel(cpFiles);
if N < 1, return, end

% load saved data and initialize list of trained models
if exist(saveFile,'file')
    load(saveFile)
end
if ~exist('csfaModels','var')
    csfaModels = {};
end

% add training data from each checkpoint file to one list
cpPath = {};
for k = 1:N
    cpPath{k} = [cpDir cpFiles(k).name];
    trainInfo = load(cpPath{k});
    
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

% delete checkpoint files once all data is saved
for k = 1:N
   delete(cpPath{k})
end
