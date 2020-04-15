function parseDeapData(saveFile)
% parseDeapData
%   parses DEAP dataset into a usable format for the rest of the CSFA codebase.
%   For dataset access, visit: http://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html
%
%   EXAMPLE: parseDeapData('dataStore')
%
%   INPUTS
%   saveFile: string inidicating name of file at which to save data. Will be saved in
%     the data directory of current folder.
%
%   SAVED VARIABLES
%   (saved to saveFile)
%   data: MxNXW array of the data for each delay length. M is the #
%       of time points. N is the # of channels. W is the # of
%       windows. All elements corresponding to data that was not
%       save (i.e. missing channel) should be marked with NaNs.
%   labels: Structure containing labeling infomation for data
%       FIELDS
%       channel: cell arrays of labels for channels
%       fs: sampling frequency of data (Hz)
%       windowLength: length of a single time window (s)
%       windows: struct array for each individual window
%           FIELDS
%           videoID: video id number
%           subjectID: participant id number
%           valence, arousal, dominance, liking: deap emotion labels

% channels to use
CHANNEL = [4 21 3 20 19 7 25 24 11 29 16 14 32];
CHANNEL_LABEL = {'F7','F8','F3','F4','Fz','C3','C4','Cz','P3', ...
    'P4','Pz','O1','O2'};
FS = 128;
PRETRIAL_SECS = 3;
WINDOW_LENGTH = 5; % seconds
DATA_DIR = 'data/';

files = dir([DATA_DIR 's*.mat']);
load([DATA_DIR files(1).name]);

% initialize things
pretrialPts = FS*PRETRIAL_SECS;
T = WINDOW_LENGTH*FS;
W = floor((size(data,3)-pretrialPts)/T);
nTrials = size(data,1);
C = numel(CHANNEL);
F = numel(files);
data_all = zeros(T, C, W);
label_all.windows.valence = zeros(W*nTrials,1);
label_all.windows.arousal = zeros(W*nTrials,1);
label_all.windows.dominance = zeros(W*nTrials,1);
label_all.windows.liking = zeros(W*nTrials,1);
label_all.windows.videoID = zeros(W*nTrials,1,'uint16');
label_all.windows.subjectID = zeros(W*nTrials,1,'uint16');

idx1 = 0; idx2 = 0;
for f = 1:F
    % load each file and store relevant data
    filename = files(f).name;
    load([DATA_DIR filename]);
    if mod(f,10) == 0
        fprintf('Processing at %d.\n',f);
    end
    
    relevantData = data(:,CHANNEL,pretrialPts+1:end);
    dataTemp = reshapeData(relevantData,T,W,nTrials);
    idx1 = idx2 + 1;
    idx2 = idx2 + size(dataTemp,3);
    data_all(:,:,idx1:idx2) = dataTemp;
    
    % save labels
    label_all.windows.valence(idx1:idx2) = repelem(labels(:,1),W);
    label_all.windows.arousal(idx1:idx2) = repelem(labels(:,2),W);
    label_all.windows.dominance(idx1:idx2) = repelem(labels(:,3),W);
    label_all.windows.liking(idx1:idx2) = repelem(labels(:,4),W);
    label_all.windows.videoID(idx1:idx2) = repelem((1:40)',W);
    label_all.windows.subjectID(idx1:idx2) = getSubjectID(filename);
    
    % clear loaded variables
    clear('data','labels');
end

labels = label_all;
data = data_all;

% fill in remaining labels
labels.fs = FS;
labels.channel = CHANNEL;
labels.windowLength = WINDOW_LENGTH;

save([DATA_DIR saveFile],'data','labels','-v7.3')
end

% reshapes data to windows and removes unwanted channels
% units: [trials,channels,time] -> [time,channels,windows]
function data = reshapeData(data,T,W,nTrials)
C = size(data,2);
data = permute(data,[3 1 2]);
data = reshape(data,[T,W*nTrials,C]);
data = permute(data,[1,3,2]);
end

% get subject id from subject datafile name
function id =  getSubjectID(filename)
idString = filename(2:3);
id = str2num(['uint16(' idString ')']);
end