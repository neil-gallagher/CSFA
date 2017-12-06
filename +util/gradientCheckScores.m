% compare numerical gradient with analytical gradient
delta = 1e-6;
tol = 1e-3;
absThresh = 1e-2;
    
y2 = y(:,:,w).'; % data for window w
y2 = y2(:); % vectorization of window w data

L = size(theseScores,1);
LLwPlus = zeros(L,1);
LLwMinus = LLwPlus;

for l = 1:L
    xPlus = theseScores(:,w); xPlus(l) = xPlus(l) + delta;
    UKUvals = sum(bsxfun(@times,UKUlStore, ...
                         permute(xPlus,[2,1])),2); % get nonzero values
    UKUclean = sparse(rows,cols,UKUvals,Nc*Ns,Nc*Ns); % put into sparse matrix
    UKUPlus = (UKUclean + 1/self.eta * speye(Nc*Ns)); % add white noise
    logDetUKUPlus = full(2*sum(log(diag(chol(2*UKUPlus)))));
    LLwPlus(l) = -Nc*Ns*log(pi) - logDetUKUPlus - (1/2)* y2'*(UKUPlus\y2);

    xMinus = theseScores(:,w); xMinus(l) = xMinus(l) - delta;
    UKUvals = sum(bsxfun(@times,UKUlStore, ...
                         permute(xMinus,[2,1])),2); % get nonzero values
    UKUclean = sparse(rows,cols,UKUvals,Nc*Ns,Nc*Ns); % put into sparse matrix
    UKUMinus = (UKUclean + 1/self.eta * speye(Nc*Ns)); % add white noise
    logDetUKUMinus = full(2*sum(log(diag(chol(2*UKUMinus)))));
    LLwMinus(l) = -Nc*Ns*log(pi) - logDetUKUMinus - (1/2)* y2'*(UKUMinus\y2);
end

% fix for all l
gradNum = real(LLwPlus - LLwMinus)/(2*delta);
thisSgrad = sgrad(:,inds(w));

gradsDiff = abs((thisSgrad-gradNum)./gradNum)>tol & abs(gradNum)>absThresh;
%gradsDiff = sign(thisSgrad) ~= sign(gradNum);
if any(gradsDiff)
    warning('Analytical and numerical gradients don''t match. A:%g N:%g\n',...
      [thisSgrad(gradsDiff),gradNum(gradsDiff)]')
end

clear y2 UKUvals UKUclean xPlus UKUPlus logDetUKUPlus LLwPlus xMinus UKUMinus
clear logDetUKUMinus LLwMinus delta tol gradNum thisSgrad

