%plotExample
MODELFILE = 'data/cvSplit1.mat';
DATAFILE = 'data/datastore.mat';
FACTOR_IDX = 3;
MODEL_IDX = 1;
CHANNEL_LABEL = {'F7','F8','F3','F4','Fz','C3','C4','Cz','P3', ...
                 'P4','Pz','O1','O2'};

% load trained model
load(MODELFILE)
model = csfaModels{MODEL_IDX}.trainModels(end);

% load frquency bounds
load(DATAFILE,'dataOpts')
minFreq = dataOpts.lowFreq;
maxFreq = dataOpts.highFreq;

% plot set of spectral densities for factor
model.plotRelativeCsd(FACTOR_IDX,'minFreq',minFreq,'maxFreq',maxFreq,'names',CHANNEL_LABEL)