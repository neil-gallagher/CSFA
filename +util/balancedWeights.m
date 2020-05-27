function weights = balancedWeights(subjectLabel, classLabel, groupLabel)
subName = unique(subjectLabel);
groupList = unique(groupLabel);
weights = zeros(length(classLabel),1);

withinSubCompare = false;
for m = subName
    mIdx = getSubIdx(subjectLabel,m);
    
    % check if multiple classes per mouse.
    if numel(unique(classLabel(mIdx))) > 1
        %If so, switch to different weighting scheme
        withinSubCompare = true;
        break
    end
    weights(mIdx) = 1/sum(mIdx);
end

if withinSubCompare
    for m = subName
        mIdx = getSubIdx(subjectLabel,m);
        
        pos = mIdx(:) & classLabel(:);
        neg = mIdx(:) & ~classLabel(:);
        
        weights(pos) = 1/sum(pos);
        weights(neg) = 1/sum(neg);
    end
else
    classList = unique(classLabel);
    for c = classList'
        cIdx = classLabel == c;
        weights(cIdx) = weights(cIdx)/sum(cIdx);
    end
end

for g = groupList
    gIdx = getSubIdx(groupLabel, g);
    weights(gIdx) = weights(gIdx)/sum(gIdx);
end

weights = weights/mean(weights);
end

function sIdx = getSubIdx(subjectLabel,m)
    if isa(subjectLabel, 'cell')
        sIdx = strcmp(subjectLabel, m{1});
    else
        sIdx = subjectLabel == m;
    end
end