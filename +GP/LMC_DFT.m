classdef LMC_DFT < GP.LMC
    % LMC_DFT   Linear model of coregionalization class with method
    % overrides for DFT approximations
    %
    % LMC_DFT Inherits
    %    LMC - properties/methods for the linear model of coregionalization
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
    
    methods
        function self = LMC_DFT(varargin)
            self = self@GP.LMC(varargin{:});
        end
        
        function res = evaluate(self,s,yfft,opts,UKUinv)
            if ~exist('opts','var') || ~isfield(opts,'Ez')
                opts.Ez = ones(1,size(yfft,3)); end
            if ~exist('UKUinvri','var'),
                opts.block = true;
                opts.smallFlag = true;
                UKUinv = self.getUKUinv(s,opts);
            end
            
            res = sum(opts.Ez .* self.logEvidence(yfft,UKUinv));
        end
        
        function res = logEvidence(self,yfft,UKUinv)
            Ns = size(yfft,1);
            
            UKUinv = 1/2*UKUinv;
            UKUinvri = 1/2*cat(5,cat(4,real(UKUinv),imag(-UKUinv)),...
                cat(4,imag(UKUinv),real(UKUinv)));
            yfftri = cat(4,real(yfft),imag(yfft));
            
            logDetTerm2 = 0;
            for n = 1:Ns
                Gammainvnri = [UKUinvri(:,:,n,1,1) UKUinvri(:,:,n,1,2); ...
                    UKUinvri(:,:,n,2,1) UKUinvri(:,:,n,2,2)];
                logDetTerm2 = logDetTerm2 + log(det(Gammainvnri));
            end
            
            Kinvy2 = sum(sum(bsxfun(@times,UKUinvri,...
                permute(yfftri,[5,2,1,6,4,3])),2),5);
            ytKinvy2 = squeeze(sum(sum(bsxfun(@times,Kinvy2,...
                permute(yfftri,[2,5,1,4,6,3])),1),4));
            
            res = -Ns*self.C/2*log(2*pi) + 1/2*logDetTerm2 - 1/2*sum(ytKinvy2,1);
        end
        
        function [kgrad,Bgrad] = deriv(self,s)
            Ns = numel(s);
            
            % store gamma and coregionalization matrix derivative pairs
            % for each parameter
            specdens = arrayfun(@(x)self.kernels.specDens(s,x),1:self.Q,'un',0);
            kgradcell = arrayfun(@(x)specdens{x},self.coregs.getKernelNums,'un',0);
            Bgradcell = self.coregs.deriv;
            
            if self.updateNoise
                kgradcell = vertcat( ones(1,Ns), kgradcell );
                Bgradcell = vertcat( -1/self.gamma^2*eye(self.C), Bgradcell);
            end
            
            % add kernel derivative pairs if necessary
            if self.updateKernels
                mats = arrayfun(@(x)self.coregs.getMat(x),1:self.Q,'un',0);
                kgradcell = vertcat( kgradcell, self.kernels.derivSpecDens(s) );
                Bgradcell = vertcat( Bgradcell, cellfun(@(x)mats{x}, ...
                    num2cell(self.kernels.getKernelNums),'un',0));
            end
            
            nParams = numel(kgradcell);
            kgrad = cat(1,kgradcell{:});
            Bgrad = cat(3,Bgradcell{:});
            kgrad = permute(kgrad,[3,4,2,1]);
            Bgrad = permute(Bgrad,[1,2,4,3]);
            
            kgrad = reshape(kgrad,[Ns,nParams]);
            Bgrad = reshape(Bgrad,[self.C^2,nParams]);
        end
        
        function grad = gradient(self,s,yfft,opts)
            if ~isfield(opts,'Ez'), opts.Ez = ones(1,size(yfft,3)); end
            if ~isfield(opts,'indepFlag'), opts.indepFlag = 0; end
            
            [Ns,Nc,~] = size(yfft);
            yfft = conj(yfft)/2;
            
            opts.block = true;
            opts.smallFlag = true;
            UKUinv = self.getUKUinv(s,opts);
            
            A = self.getA(yfft,opts.Ez,UKUinv);
            A = reshape(A,[Nc^2,Ns]);
            
            [kgrad,Bgrad] = self.deriv(s);
            
            grad = (A'*Bgrad).*kgrad;
            grad = sum(real(grad),1)';
        end
        
        function res = getUKUinv(self,s,opts)
            if ~isfield(opts,'UKU')
                opts2 = opts;
                opts2.smallFlag = false;
                UKU = self.UKU(s,opts2);
            else
                UKU = opts.UKU;
            end
            
            if self.C == 1
                res = 1./UKU;
            else
                [Qd,Rd] = qr(UKU); % qr decomposition
                res = Qd/(Rd');    % inverse using qr
            end
            
            if isfield(opts,'smallFlag') && opts.smallFlag
                res = self.extractBlocks(res); end
        end
        
        function y = sample(self,N,delta)
            Ns = ceil(N/2);
            s = 1/(N*delta):1/(N*delta):1/(2*delta);
            
            opts.block = true;
            UKU = self.extractBlocks(self.UKU(s,opts));
            Uy = zeros(Ns,self.C);
            for n = 1:Ns
                UKUn = UKU(:,:,n);
                UKUnri = 2*[real(UKUn) imag(-UKUn); imag(UKUn) real(UKUn)];
                Uyri = mvnrnd(zeros(2*self.C,1),UKUnri);
                Uy(n,:) = Uyri(1:self.C) + 1i*Uyri(self.C+1:end);
            end
            y = real(ifft([Uy; flipud(Uy)]));
        end
    end
    
    methods(Static)
        function Akeep = getA(yfft,Ez,UKUinv)
            % get matrix A = a*a' - W*K^{-1} for GP gradient steps
            
            data = bsxfun(@times,yfft,reshape(Ez,[1,1,numel(Ez)]));
            W = sum(Ez);
            
            alpha1 = sum(bsxfun(@times,UKUinv,permute((data),[4,2,1,3])),2);
            alpha2 = sum(bsxfun(@times,UKUinv,permute(conj(data),[2,4,1,3])),1);
            
            Akeep = sum(bsxfun(@times,alpha1,alpha2),4) - W*UKUinv;
        end
    end
end