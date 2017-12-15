function initCvFiles(datafile)
% initCvFiles
% initializes model files for k-fold test sets
%
%   INPUT
%   datafile: name of file containing data saved from preprocessDeap function
%
%   SAVED VARIABLES
%   sets: structure giving train/val/test split
%       FIELDS
%       train: logical vector indicating windows in xFft to be used
%           in training set
%       val(optional): logical vector indicating window to be used in
%           validation
%       datafile: path to file containing data used to train model
%       test (optional): logical vector indicating windows for
%           testing
%       description (optional): describes validation set scheme


N_SUBJECTS = 32;
N_TEST = 8;

DATA_FOLDER = 'data/';
filepath = [DATA_FOLDER datafile];

K = N_SUBJECTS/N_TEST;
load(filepath,'labels')

randId = randperm(N_SUBJECTS);
sId = labels.windows.subjectID;

sets.datafile = filepath;
for k = 1:K
  idx = ((k-1)*N_TEST + 1):k*N_TEST;
  sets.test = ismember(sId,randId(idx));
  sets.train = ~sets.test;

  modFile = sprintf('data/cvSplit%d.mat',k);
  save(modFile,'sets')
end