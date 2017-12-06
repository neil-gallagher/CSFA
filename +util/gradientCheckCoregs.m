% compare numerical gradient with analytical gradient
delta = 1e-6;
tol = 1e-3;
absThresh = 1e-2;
opts.block = true;
opts.smallFlag = false;

L = size(self.scores,1);
P = self.LMCkernels{1}.coregs.nParams;
gradNum = zeros(P,L,sum(self.W));

for p = 1:Parts

  % extract data and factors scores for partition p
  if ~stochastic
    inds = ((p-1)*self.maxW+1):((p-1)*self.maxW+self.W(p));
    y = data(modelFreqs,:,inds);
  else
    y = data(modelFreqs,:,:); 
  end
  y = conj(y);
  theseScores = self.scores(:,inds);

  for w = 1:numel(inds)
    
    yThisWin = y(:,:,w).'; % data for window w
    yThisWin = yThisWin(:); % vectorization of window w data

    LLwPlus = zeros(P,L);
    LLwMinus = LLwPlus;

    for l = 1:size(theseScores,1)
      Kl = self.LMCkernels{l}.copy;
      
      for q = 1:self.Q
        Bq = Kl.coregs.B{q};
        for r = 1:Bq.R
          for c = 1:Bq.C
            % get LL w/ increment in parameter
            Bq.logWeights(c,r) = ...
                self.LMCkernels{l}.coregs.B{q}.logWeights(c,r) + delta;
            [~,UKUlPlus] = Kl.UKU(s,opts); % get factor l Gram matrix
            UKUlVals = Kl.extractBlocks(UKUlPlus);
            UKUlStorePlus = UKUlStore; UKUlStorePlus(:,l) = UKUlVals(:);
            UKUvals = UKUlStorePlus*theseScores(:,w);
            UKUPlus = sparse(rows,cols,UKUvals,Nc*Ns,Nc*Ns); % put into sparse matrix
            UKUPlus = UKUPlus + 1/self.eta * speye(Nc*Ns); % add white noise
            UKUPlus = 2*UKUPlus;
            logDetUKUPlus = full(2*sum(log(diag(chol((UKUPlus + UKUPlus')./2)))));
            paramNo = (q-1)*Bq.nParams + (r-1)*Bq.C + c;
            LLwPlus(paramNo,l) = -Nc*Ns*log(pi) - logDetUKUPlus - yThisWin'*(UKUPlus\yThisWin);

            Bq.logWeights(c,r) = ...
                self.LMCkernels{l}.coregs.B{q}.logWeights(c,r);
            
            % LL w/ decrement
            Bq.logWeights(c,r) = ...
                self.LMCkernels{l}.coregs.B{q}.logWeights(c,r) - delta;
            [~,UKUlMinus] = Kl.UKU(s,opts); % get factor l Gram matrix
            UKUlVals = Kl.extractBlocks(UKUlMinus);
            UKUlStoreMinus = UKUlStore; UKUlStoreMinus(:,l) = UKUlVals(:);
            UKUvals = UKUlStoreMinus*theseScores(:,w);
            UKUMinus = sparse(rows,cols,UKUvals,Nc*Ns,Nc*Ns); % put into sparse matrix
            UKUMinus = UKUMinus + 1/self.eta * speye(Nc*Ns); % add white noise
            UKUMinus = 2*UKUMinus;
            logDetUKUMinus = full(2*sum(log(diag(chol((UKUMinus + UKUMinus')./2)))));
            paramNo = (q-1)*Bq.nParams + (r-1)*Bq.C + c;
            LLwMinus(paramNo,l) = -Nc*Ns*log(pi) - logDetUKUMinus - yThisWin'*(UKUMinus\yThisWin);

            Bq.logWeights(c,r) = ...
                self.LMCkernels{l}.coregs.B{q}.logWeights(c,r);
          end
          
          % shifts
          for c = 2:Bq.C
            % get LL w/ increment in parameter
            Bq.shifts(c,r) = ...
                self.LMCkernels{l}.coregs.B{q}.shifts(c,r) + delta;
            [~,UKUlPlus] = Kl.UKU(s,opts); % get factor l Gram matrix
            UKUlVals = Kl.extractBlocks(UKUlPlus);
            UKUlStorePlus = UKUlStore; UKUlStorePlus(:,l) = UKUlVals(:);
            UKUvals = UKUlStorePlus*theseScores(:,w);
            UKUPlus = sparse(rows,cols,UKUvals,Nc*Ns,Nc*Ns); % put into sparse matrix
            UKUPlus = UKUPlus + 1/self.eta * speye(Nc*Ns); % add white noise
            UKUPlus = 2*UKUPlus;
            logDetUKUPlus = full(2*sum(log(diag(chol((UKUPlus + UKUPlus')./2)))));
            paramNo = (q-1)*Bq.nParams + (r-1)*(Bq.C-1) + Bq.R*Bq.C + c-1;
            LLwPlus(paramNo,l) = -Nc*Ns*log(pi) - logDetUKUPlus - yThisWin'*(UKUPlus\yThisWin);

            Bq.shifts(c,r) = ...
                self.LMCkernels{l}.coregs.B{q}.shifts(c,r);
            
            % LL w/ decrement
            Bq.shifts(c,r) = ...
                self.LMCkernels{l}.coregs.B{q}.shifts(c,r) - delta;
            [~,UKUlMinus] = Kl.UKU(s,opts); % get factor l Gram matrix
            UKUlVals = Kl.extractBlocks(UKUlMinus);
            UKUlStoreMinus = UKUlStore; UKUlStoreMinus(:,l) = UKUlVals(:);
            UKUvals = UKUlStoreMinus*theseScores(:,w);
            UKUMinuLLwPlus(paramNo,l) = -Nc*Ns*log(pi) - logDetUKUPlus - yThisWin'*(UKUPlus\yThisWs = sparse(rows,cols,UKUvals,Nc*Ns,Nc*Ns); % put into sparse matrix
            UKUMinus = UKUMinus + 1/self.eta * speye(Nc*Ns); % add white noise
            UKUMinus = 2*UKUMinus;
            logDetUKUMinus = full(2*sum(log(diag(chol((UKUMinus + UKUMinus')./2)))));
            paramNo = (q-1)*Bq.nParams + (r-1)*(Bq.C-1) + Bq.R*Bq.C + c-1;
            LLwMinus(paramNo,l) = -Nc*Ns*log(pi) - logDetUKUMinus - yThisWin'*(UKUMinus\yThisWin);

            Bq.shifts(c,r) = ...
                self.LMCkernels{l}.coregs.B{q}.shifts(c,r);
          end
        end
      end
    end

    gradNum(:,:,inds(w)) = real(LLwPlus - LLwMinus)/(2*delta);
    
  end
end

gradNum = sum(gradNum,3);
gradsDiff = abs((LMCgrad(1:P,:)-gradNum)./gradNum)>tol & abs(gradNum)>absThresh;
if any(gradsDiff(:))
  warning('Analytical and numerical gradients don''t match. A:%f N:%f\n',...
          [LMCgrad(gradsDiff),gradNum(gradsDiff)]')
end



