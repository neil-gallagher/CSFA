# csm
Cross-Spectral Mixture Model Code

The main functions for use in this repository are trainCSFA.m, projectCSFA.m, and saveTrainRuns.m
First see the header documentation in those functions.

trainCSFA expects a file containing the sets variable, which is a structure defining a train/val/test split of the data. The function saves each model in an intermediate checkpoint file that begins with 'chkpt'.
saveTrainRuns saves all of the intermediate checkpoint files with the same train/val/test split in the same file.

