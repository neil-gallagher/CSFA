function [selection,p] = selectDiscFactors(numFactors,target,scores)
% Select Discriminatory Factors
% This takes the CSFA generative model created in
% trainCSFA and finds the factors with the most
% signifciant different between two conditions. Returns indexes of
% most significant factors in a logical array. Significance
% determined using Wilcoxin rank sum test.
% Example:
% dIdx =
% selectDiscFactors(3,labels.windows.genotype(sets.train),scores(sets.train));
%
% Thanks to Kat Hefter for writing the first draft of this function!

L = size(scores,1);

if isa(target, 'cell')
    target = categorical(target);
end

% get unique identifiers for all classes. consider binary case as one class
uniqueClasses = unique(target);
C = numel(uniqueClasses);
if C == 2
    uniqueClasses = uniqueClasses(2);
    C = 1;
end

% for each class, rank factors based on rank-sum p values
p = zeros(L,C);
for c = 1:C
    thisClassLabel = target == uniqueClasses(c);
    
    for s = 1:L
        p(s, c) = ranksum(scores(s, thisClassLabel), scores(s, ~thisClassLabel));
    end
end

avgLogP = mean(log(p),2,'omitnan');
[~,sortingIndices] = sort(avgLogP,'ascend');
indx = sortingIndices(1:numFactors);
selection = false(L,1);
selection(indx) = true;

end
