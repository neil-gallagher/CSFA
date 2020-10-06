% compare numerical gradient with analytical gradient
delta = 1e-6;
tol = 1e-3;
absThresh = 1e-2;

L = size(features,1);
HLwPlus = zeros(L,1);
HLwMinus = HLwPlus;

for l = 1:L
    xPlus = features'; xPlus(:,l) = xPlus(:,l) + delta;
    mPlus = margin(cmodel,xPlus,y)/2;
    HLwPlus = max(0, 1-mPlus);

    xMinus = features'; xMinus(:,l) = xMinus(:,l) - delta;
    mMinus = margin(cmodel,xMinus,y)/2;
    HLwMinus = max(0, 1-mMinus);
end

% fix for all l
gradNum = real(HLwPlus - HLwMinus)/(2*delta);

gradsDiff = abs((gradHL-gradNum)./gradNum)>tol & abs(gradHL-gradNum)>absThresh;
%gradsDiff = sign(thisSgrad) ~= sign(gradNum);

% accound for examples on boundary
bound = (mMinus-1)<tol | (mPlus-1)<tol;
gradsDiff(bound) = false;

if any(gradsDiff)
    warning('Analytical and numerical gradients don''t match.')
end

clear xPlus mPlus HLwPlus xMinus mMinus HLwMinus delta tol gradNum thisSgrad

