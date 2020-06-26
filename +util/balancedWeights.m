function weights = balancedWeights(groupList, subsetIdx)
% For balancing sample weights. group_list should be a list of group
% labels. The first grouping listed wil be balancde across the whole
% dataset. The next grouping listed will be relatively balanced within each
% individual group given by the first grouping. That pattern continues
% iterativel so that each grouping listed is relatively balanced within
% individual groups from the grouping that comes before it in the list.

% if subsetIdx not given, set to all windows
if nargin < 2
    subsetIdx = true(1,length(groupList{1}));
end

% get list of labels balanced in this recursive iteration
groupLabels = groupList{1};
theseGroupLabels = groupLabels(subsetIdx);

% get unique groups
groupNames = unique(theseGroupLabels);

weights = ones(1, sum(subsetIdx));
% for each unique group, get all associated indicies
for g = 1:numel(groupNames)
    if isa(groupLabels, 'cell')
        gIdx = strcmp(groupLabels, groupNames{g});
    else
        gIdx = groupLabels == groupNames(g);
    end
    gSubIdx = gIdx(subsetIdx);
    
    % if there are remaining groups, get those relative weights, then apply
    % this weight
    if numel(groupList) > 1
       subgroupWeights = util.balancedWeights(groupList(2:end), subsetIdx(:)&gIdx(:));
       weights(gSubIdx) = subgroupWeights ./ sum(gSubIdx);
    else
        % otherwise just apply this weight
        weights(gSubIdx) = 1/sum(gSubIdx);
    end
end

% normalize to have mean of 1
weights = weights./mean(weights);
end