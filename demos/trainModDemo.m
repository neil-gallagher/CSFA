function trainMod(description,dataFile,modelFile)

mOpts.Q = 3;
mOpts.L = 21;
mOpts.R = 2;
mOpts.eta = 10;
mOpts.description = description;

rng('shuffle')

trainCSFA(dataFile,modelFile,mOpts)

end