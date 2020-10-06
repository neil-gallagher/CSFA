% compare numerical gradient with analytical gradient
delta = 1e-6;
tol = 1e-3;
absThresh = 1e-2;

L = size(features,1)-1;
CEwPlus = zeros(L,1);
CEwMinus = CEwPlus;

for l = 1:L
    xPlus = features'; xPlus(:,l) = xPlus(:,l) + delta;
    pPlus = predict(self.classModel,xPlus);
    CEwPlus = -thisLabel.*log(pPlus) - (1-thisLabel).*log(1-pPlus);

    xMinus = features'; xMinus(:,l) = xMinus(:,l) - delta;
    pMinus = predict(self.classModel,xMinus);
    CEwMinus = -thisLabel.*log(pMinus) - (1-thisLabel).*log(1-pMinus);
end

% fix for all l
gradNum = real(CEwPlus - CEwMinus)'/(2*delta);

gradsDiff = abs((gradCE-gradNum)./gradNum)>tol & abs(gradCE-gradNum)>absThresh;
%gradsDiff = sign(thisSgrad) ~= sign(gradNum);

if any(gradsDiff)
    warning('Analytical and numerical gradients don''t match.')
end

clear xPlus mPlus CEwPlus xMinus mMinus CEwMinus delta tol gradNum thisSgrad

