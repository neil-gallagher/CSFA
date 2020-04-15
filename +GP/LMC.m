classdef LMC < handle & GP.spectrumPlots
    % LMC   Linear model of coregionalization class
    %
    % LMC Inherits
    %    spectrumPlots - cross-spectrum plotting functions
    %
    % LMC Methods (of interest):
    %    evaluate - evaluate the log evidence of a batch of observations
    %    gradient - obtain the gradient vector wrt all kernel parameters
    %    K - obtain the full kernel Gram matrix
    %    UKU - obtain the Fourier transform of the Gram matrix
    %    sample - sample data
    %    plotCsd - Plot all cross-spectra
    %    plotCsdComp - Plot single cross-spectrum component
    
    properties
        % model parameters
        C % number of observation channels
        Q % number of kernels
        
        kernels % GP basis kernels
        coregs % coregionalization matrices
        gamma % white noise precision
        
        updateKernels % provide updates to kernel params during learning
        updateNoise % provide updates to noise param during learning
    end
    
    methods
        function self = LMC(C,Q,R,coregs,kernels,gamma)
            if nargin > 0
                self.C = C;
                self.Q = Q;
                
                if ~exist('coregs','var'), self.coregs = GP.coregs.mats(C,Q,R);
                else
                    self.coregs = coregs;
                end
                
                if ~exist('kernels','var'), self.kernels = GP.kernels.kernels(Q);
                else
                    self.kernels = kernels;
                end
                
                if ~exist('gamma','var'), self.gamma = 1000;
                else
                    self.gamma = gamma;
                end
                
                self.updateKernels = 1;
                self.updateNoise = 1;
            end
        end
        
        function res = getParams(self)
            res = self.coregs.getParams;
            if self.updateNoise
                res = [self.gamma; res]; end
            if self.updateKernels
                res = [res; self.kernels.getParams]; end
        end
        
        function res = nParams(self)
            res = numel(self.getParams);
        end
        
        function setParams(self,params)
            ind = 1;
            if self.updateNoise
                self.gamma = params(1); ind = ind+1; end
            nParams = self.coregs.nParams;
            self.coregs.setParams(params(ind:1+nParams));
            ind = ind+nParams;
            if self.updateKernels
                self.kernels.setParams(params(ind:end)); end
        end
        
        function [lb,ub] = getBounds(self)
            [lbk,ubk] = self.kernels.getBounds;
            [lbc,ubc] = self.coregs.getBounds;
            lb = lbc; ub = ubc;
            if self.updateNoise
                lb = [1e-1; lb]; ub = [1e4; ub]; end
            if self.updateKernels
                lb = [lb; lbk]; ub = [ub; ubk]; end
        end
        
        function res = evaluate(self,tau,y,opts)
            if ~exist('opts','var') || ~isfield(opts,'Ez')
                opts.Ez = ones(1,size(y,3)); end
            
            res = sum(opts.Ez .* self.logEvidence(tau,y));
        end
        
        function res = logEvidence(self,tau,y)
            K = real(self.K(tau));
            
            logdetK = 2*sum(log(diag(chol(K))));
            N = numel(tau); W = size(y,3);
            y = reshape(y,[N*self.C,W]);
            
            res = -1/2*N*self.C*log(2*pi) - 1/2*logdetK;
            res = res - 1/2*sum(y.*(K\y),1);
        end
        
        function [kgrad,Bgrad] = deriv(self,tau)
            N = numel(tau);
            
            % store gamma and coregionalization matrix derivative pairs
            % for each parameter
            covFuns = arrayfun(@(x)self.kernels.covFun(tau,x),1:self.Q,'un',0);
            kgradcell = arrayfun(@(x)covFuns{x},self.coregs.getKernelNums,'un',0);
            Bgradcell = self.coregs.deriv;
            
            if self.updateNoise
                kgradcell = vertcat( ones(1,N), kgradcell );
                Bgradcell = vertcat( -1/self.gamma^2*eye(self.C), Bgradcell);
            end
            
            % add kernel derivative pairs if necessary
            if self.updateKernels
                mats = arrayfun(@(x)self.coregs.getMat(x),1:self.Q,'un',0);
                kgradcell = vertcat( kgradcell, self.kernels.derivCovFun(tau) );
                Bgradcell = vertcat( Bgradcell, cellfun(@(x)mats{x}, ...
                    num2cell(self.kernels.getKernelNums),'un',0));
            end
            
            nParams = numel(kgradcell);
            kgrad = cat(1,kgradcell{:});
            Bgrad = cat(3,Bgradcell{:});
            kgrad = permute(kgrad,[3,4,2,1]);
            Bgrad = permute(Bgrad,[1,2,4,3]);
            
            kgrad = reshape(kgrad,[N,nParams]);
            Bgrad = reshape(Bgrad,[self.C,self.C,nParams]);
        end
        
        function grad = gradient(self,tau,y,opts)
            if ~isfield(opts,'Ez'), opts.Ez = ones(1,size(y,3)); end
            if ~isfield(opts,'indepflag'), opts.indepflag = 0; end
            
            W = size(y,3);
            N = size(y,1);
            y = reshape(y,[N*self.C,W]);
            
            K = real(self.K(tau));% get covariance matrix
            [Qd,Rd] = qr(K);      % qr decomposition
            Kinv = Qd/(Rd');      % inverse using qr decomposition
            
            alpha = Kinv*y;
            A = alpha*alpha' - W*Kinv;
            At = A'; At = At(:)';
            
            [kgrad,Bgrad] = self.deriv(tau);
            nParams = size(kgrad,2);
            
            grad = zeros(nParams,1);
            for p = 1:nParams
                dK = real( kron( Bgrad(:,:,p), toeplitz(kgrad(:,p)) ) );
                grad(p) = 1/2*(At*dK(:));
            end
        end
        
        function [resNoise,resNoNoise] = K(self,tau)
            N = numel(tau);
            resNoNoise = zeros(self.C*N);
            for q = 1:self.Q
                B = self.coregs.getMat(q);
                k = toeplitz(self.kernels.covFun(tau,q));
                resNoNoise = resNoNoise + kron(B,k);
            end
            resNoise = resNoNoise + 1/self.gamma*eye(numel(tau)*self.C);
        end
        
        function res = covFuns(self,tau)
            N = numel(tau);
            res = zeros(N,self.C);
            for q = 1:self.Q
                B = self.coregs.getMat(q);
                k = real(self.kernels.covFun(tau,q))';
                res = res + bsxfun(@times,diag(real(B))',k);
            end
            res(1,:) = res(1,:) + 1/self.gamma; % additive white noise
        end
        
        function maxCov = normalizeCovariance(self)
            maxCov = max(self.covFuns(0));
            for q = 1:self.Q
                self.coregs.B{q}.logWeights =  ...
                    self.coregs.B{q}.logWeights - 1/2*log(maxCov);
            end
            self.gamma = self.gamma*maxCov;
        end
        
        function [resNoise,resNoNoise] = UKU(self,s,opts)
            Ns = numel(s); d = 1/(2*Ns*(s(2)-s(1)));
            Nc = self.C;
            
            SD = self.kernels.specDens(s);
            res = spalloc(Ns*Nc,Ns*Nc,Nc^2*Ns);
            
            for q = 1:self.Q
                if isfield(opts,'Bmats') && ~isempty(opts.Bmats)
                    B = opts.Bmats{q};
                else
                    B = self.coregs.getMat(q);
                end
                
                if isfield(opts,'indepFlag') && opts.indepFlag
                    B = diag(diag(B)); end
                
                if isfield(opts,'block') && opts.block
                    res = res + 1/d/2*kron(spdiags(SD(:,q),0,Ns,Ns),B);
                else
                    res = res + 1/d/2*kron(B,spdiags(SD(:,q),0,Ns,Ns));
                end
            end
            
            resNoise = res + spdiags(1/self.gamma*ones(Ns*Nc,1),0,Ns*Nc,Ns*Nc);
            resNoNoise = res;
            
            if isfield(opts,'smallFlag') && opts.smallFlag
                resNoise = self.extractBlocks(resNoise);
                resNoNoise = self.extractBlocks(resNoNoise);
            end
        end
        
        function y = sample(self,N,delta)
            tau = 0:delta:(N-1)*delta;
            K = real(self.K(tau));
            
            y = chol(K)*randn(N*self.C,1);
            y = reshape(y,[N,self.C]);
        end
        
        function [res,rows,cols] = extractBlocks(self,mat)
            NC = size(mat,1);
            N = NC/self.C;
            inds = logical(kron(speye(N),ones(self.C)));
            [rows,cols] = find(inds);
            fMat = full(mat(inds));
            res = reshape(fMat,[self.C,self.C,N]);
        end
        
        function plotCsd(self,varargin)
            % plot the cross-spectral density of the kernel.  Inputs are
            % specified param-value pairs.
            
            p = inputParser;
            p.KeepUnmatched = true;
            addOptional(p,'minFreq',0,@isnumeric);
            addOptional(p,'maxFreq',30,@isnumeric);
            addOptional(p,'Bmats',[])
            parse(p,varargin{:});
            opts = p.Results;
            
            s = linspace(opts.minFreq,opts.maxFreq,1e3);
            opts.smallFlag = true;
            opts.block = true;
            UKU = self.UKU(s,opts);
            
            plotCsd@GP.spectrumPlots(s,UKU,varargin{:})
        end
        
        function ax = plotCsdComp(self,varargin)
            % plot a single cross-spectrum between two channels.  Inputs
            % are specified param-value pairs.
            
            p = inputParser;
            p.KeepUnmatched = true;
            addOptional(p,'minFreq',0,@isnumeric);
            addOptional(p,'maxFreq',30,@isnumeric);
            parse(p,varargin{:});
            opts = p.Results;
            
            s = linspace(opts.minFreq,opts.maxFreq,1e3);
            opts.smallFlag = true;
            opts.block = true;
            UKU = self.UKU(s,opts);
            
            ax = plotCsdComp@GP.spectrumPlots(s,UKU,varargin{:});
        end
        
        function keepChannels(self,chans)
            % replace all properties with only the channels in chans
            % NOTE: this will remove data!
            self.C = numel(chans);
            self.coregs.keepChannels(chans);
        end
        
        % Make a copy of a handle object.
        function new = copy(self)
            % Instantiate new object of the same class.
            new = feval(class(self));
            
            % Copy all non-hidden properties.
            p = properties(self);
            for i = 1:length(p)
                if ismethod(self.(p{i}),'copy')
                    new.(p{i}) = self.(p{i}).copy;
                else
                    new.(p{i}) = self.(p{i});
                end
            end
        end
    end
end