% compare numerical gradient with analytical gradient
delta = 1e-6;
tol = 1e-2;
absThresh = 1e-3;
opts.block = true;
opts.smallFlag = false;

y2 = y(:,:,w).'; % data for window w
y2 = y2(:); % vectorization of window w data

L = self.L;
P = self.LMCkernels{1}.kernels.nParams;
LLwPlus = zeros(P,L);
LLwMinus = LLwPlus;

for l = 1:L
    for q = 1:self.Q
        Kl = self.LMCkernels{l}.copy;
        
        % get LL w/ increment in parameter
        Kl.kernels.k{q}.mu = self.LMCkernels{l}.kernels.k{q}.mu + delta;
        
        [~,UKUlPlus] = Kl.UKU(s,opts); % get factor l Gram matrix
        vals3 = Kl.extractBlocks(UKUlPlus);
        UKUlStorePlus = UKUlStore; UKUlStorePlus(:,l) = vals3(:);
        vals2 = sum(bsxfun(@times,UKUlStorePlus, ...
            permute(theseScores(:,w),[2,1])),2); % get nonzero values
        UKUPlus = sparse(rows,cols,vals2,Nc*Ns,Nc*Ns); % put into sparse matrix
        UKUPlus = UKUPlus + noise;
        UKUPlus = 2*UKUPlus;
        logDetUKUPlus = full(2*sum(log(diag(chol(UKUPlus)))));
        paramNo = q*2 - 1;
        LLwPlus(paramNo,l) = -Nc*Ns*log(pi) - logDetUKUPlus - y2'*(UKUPlus\y2);
        Kl.kernels.k{q}.mu = self.LMCkernels{l}.kernels.k{q}.mu;
        
        Kl.kernels.k{q}.logVar = self.LMCkernels{l}.kernels.k{q}.logVar + delta;
        [~,UKUlPlus] = Kl.UKU(s,opts); % get factor l Gram matrix
        vals3 = Kl.extractBlocks(UKUlPlus);
        UKUlStorePlus = UKUlStore; UKUlStorePlus(:,l) = vals3(:);
        vals2 = sum(bsxfun(@times,UKUlStorePlus, ...
            permute(theseScores(:,w),[2,1])),2); % get nonzero values
        UKUPlus = sparse(rows,cols,vals2,Nc*Ns,Nc*Ns); % put into sparse matrix
        UKUPlus = UKUPlus + noise;
        UKUPlus = 2*UKUPlus;
        logDetUKUPlus = full(2*sum(log(diag(chol(UKUPlus)))));
        paramNo = q*2;
        LLwPlus(paramNo,l) = -Nc*Ns*log(pi) - logDetUKUPlus - y2'*(UKUPlus\y2);
        Kl.kernels.k{q}.logVar = self.LMCkernels{l}.kernels.k{q}.logVar;
        
        % LL w/ decrement
        Kl.kernels.k{q}.mu = self.LMCkernels{l}.kernels.k{q}.mu - delta;
        [~,UKUlMinus] = Kl.UKU(s,opts); % get factor l Gram matrix
        vals3 = Kl.extractBlocks(UKUlMinus);
        UKUlStoreMinus = UKUlStore; UKUlStoreMinus(:,l) = vals3(:);
        vals2 = sum(bsxfun(@times,UKUlStoreMinus, ...
            permute(theseScores(:,w),[2,1])),2); % get nonzero values
        UKUMinus = sparse(rows,cols,vals2,Nc*Ns,Nc*Ns); % put into sparse matrix
        UKUMinus = UKUMinus + noise;
        UKUMinus = 2*UKUMinus;
        logDetUKUMinus = full(2*sum(log(diag(chol(UKUMinus)))));
        paramNo = q*2 - 1;
        LLwMinus(paramNo,l) = -Nc*Ns*log(pi) - logDetUKUMinus - y2'*(UKUMinus\y2);
        Kl.kernels.k{q}.mu = self.LMCkernels{l}.kernels.k{q}.mu;
        
        Kl.kernels.k{q}.logVar = self.LMCkernels{l}.kernels.k{q}.logVar - delta;
        [~,UKUlMinus] = Kl.UKU(s,opts); % get factor l Gram matrix
        vals3 = Kl.extractBlocks(UKUlMinus);
        UKUlStoreMinus = UKUlStore; UKUlStoreMinus(:,l) = vals3(:);
        vals2 = sum(bsxfun(@times,UKUlStoreMinus, ...
            permute(theseScores(:,w),[2,1])),2); % get nonzero values
        UKUMinus = sparse(rows,cols,vals2,Nc*Ns,Nc*Ns); % put into sparse matrix
        UKUMinus = UKUMinus + noise;
        UKUMinus = 2*UKUMinus;
        logDetUKUMinus = full(2*sum(log(diag(chol(UKUMinus)))));
        paramNo = q*2;
        LLwMinus(paramNo,l) = -Nc*Ns*log(pi) - logDetUKUMinus - y2'*(UKUMinus\y2);
        Kl.kernels.k{q}.logVar = self.LMCkernels{l}.kernels.k{q}.logVar;
    end
end

% fix for all l
gradNum = real(LLwPlus - LLwMinus)/(2*delta);
thisKgrad2 = gather(thisKgrad(end-P+1:end,:));

%gradsDiff = sign(thisKgrad2) ~= sign(gradNum);
gradsDiff = abs((thisKgrad2-gradNum)./gradNum)>tol & abs(gradNum)>absThresh;
if any(gradsDiff(:))
    warning('Analytical and numerical gradients don''t match. A:%f N:%f\n',...
        [thisKgrad2(gradsDiff),gradNum(gradsDiff)]')
end

clear y2 vals2 xPlus UKUPlus logDetUKUPlus LLwPlus xMinus UKUMinus
clear logDetUKUMinus LLwMinus delta tol gradNum thisKgrad2

