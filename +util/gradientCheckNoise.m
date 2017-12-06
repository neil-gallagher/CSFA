% compare numerical gradient with analytical gradient
delta = 1e-6;
tol = 1e-2;

    
if isobject(data)||isstruct(data), yP = conj(data.y(:,:,inds)); % extract data for partition p if structure is provided
else yP = data(:,:,inds); end % extract data for partition p if array is provided
y2 = yP(:,:,w).'; % data for window w
y2 = y2(:); % vectorization of window w data

% yW = y(:,:,w);
% yW = yW(:);

vals2 = sum(bsxfun(@times,UKUlStore, ...
    permute(scoresAllMem(:,w),[2,1])),2); % get nonzero values
UKU2 = sparse(rows,cols,vals2,Nc*Ns,Nc*Ns); % put into sparse matrix

xPlus = gam + delta;
UKUPlus = UKU2 + 1/xPlus * speye(Nc*Ns); % add white noise
UKUPlus = 2*UKUPlus;
logDetUKUPlus = full(2*sum(log(diag(chol(UKUPlus)))));
LLwPlus = -Nc*Ns*log(pi) - logDetUKUPlus - y2'*(UKUPlus\y2);

xMinus = gam - delta;
UKUMinus = UKU2 + 1/xMinus * speye(Nc*Ns); % add white noise
UKUMinus = 2*UKUMinus;
logDetUKUMinus = full(2*sum(log(diag(chol(UKUMinus)))));
LLwMinus = -Nc*Ns*log(pi) - logDetUKUMinus - y2'*(UKUMinus\y2);

% fix for all l
gradNum = real(LLwPlus - LLwMinus)/(2*delta);
thisNgrad = gather(thisNgrad);
if abs(thisNgrad-gradNum)>tol
    warning('Analytical and numerical gradients don''t match. A:%f N:%f\n',thisNgrad,gradNum)
end

clear yP y2 vals2 UKU2 xPlus UKUPlus logDetUKUPlus LLwPlus xMinus UKUMinus
clear logDetUKUMinus LLwMinus delta tol gradNum thisNgrad

