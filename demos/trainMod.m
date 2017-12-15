function trainMod(dataFile,modelFile)
% trainMod
% trains a csfa model or models
%   INPUTS
%   datafile: string giving name of file containing data saved from
%     preprocessDeap function
%   modelFile: string giving name of file to which model(s) will be saved. To
%     set a train/val/test split this file should be initialized with a
%     structure named sets.
%     CONTAINS
%     sets: (optional) structure giving train/val/test split
%       FIELDS
%       train: logical vector indicating windows in xFft to be used
%           in training set
%       val(optional): logical vector indicating window to be used in
%           validation
%       datafile: path to file containing data used to train model
%       test (optional): logical vector indicating windows for
%           testing
%       description (optional): describes validation set scheme


mOpts.description = 'default';
mOpts.eta = 5;

trainOpts.iters = 500;
trainOpts.saveInterval = 100;

rng('shuffle')

trainCSFA(dataFile,modelFile,mOpts,trainOpts)

end