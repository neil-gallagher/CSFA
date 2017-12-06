function oneHotLabels = oneHot(labels,valueLabels)

% Takes a vector of size n by 1 as input and creates a one-hot encoding of its
% elements.
if nargin < 2
    valueLabels = unique(labels);
end
nLabels = length(valueLabels);
nSamples = numel(labels);

oneHotLabels = zeros(nLabels, nSamples);

if iscellstr(labels)
  for i = 1:nLabels
    oneHotLabels(i,:) = strcmp(labels, valueLabels(i));
  end
else
  for i = 1:nLabels
    oneHotLabels(i,:) = (labels == valueLabels(i));
  end
end

oneHotLabels = logical(oneHotLabels);