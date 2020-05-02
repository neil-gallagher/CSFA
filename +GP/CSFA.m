classdef CSFA < handle & GP.spectrumPlots
    properties
        % model parameters
        P % number of partitions of data
        W % number of time windows (observations) per partition
        maxW % maximum number of time windows in a partition
        C % number of channels
        Q % number of components in SM kernel
        L % rank of FA
        scores % factor scores
        freqBounds % [low, high] frequency boundary for spectral gaussian means
        
        LMCkernels % cell array of L LMC kernels
        
        eta % additive Gaussian noise
        
        updateKernels % biniary indicator for kernel updates
        updateScores % biniary indicator for score updates
        updateNoise % biniary indicator for noise floor updates
    end
    
    methods
        function self = CSFA(modelOpts, s, xFft)
            GMM_MAX_ITER = 1000;
            DIST_PRECISION = 1000;
            DIST_SAMPLES = 1e5;
            
            if nargin > 0
                VAR_LB = 0.5^2; VAR_UB = 25^2; % spectral guassian variance bounds
                lowF = modelOpts.lowFreq; highF = modelOpts.highFreq;
                
                % set model parameters
                self.L = modelOpts.L;
                self.C = modelOpts.C;
                self.Q = modelOpts.Q;
                self.eta = modelOpts.eta;
                
                % divide windows into partitions for computations
                self.setPartitions(modelOpts.W, modelOpts.maxW);
                
                % create and initialize L LMC kernels
                self.LMCkernels = cell(self.L,1);
                self.freqBounds = [lowF highF];
                
                if nargin < 3
                    % initialize factor scores
                    self.scores = rand(self.L,modelOpts.W);
                    
                    for l = 1:self.L
                        % initialize set of SG kernels
                        % mean and var are initialized w/ higher probability of lower
                        % values, since we thing lower frequencies and narrower bands are
                        % more likely to show up
                        means = num2cell(self.logRand(lowF,highF,self.Q));
                        vars = num2cell(self.logRand(VAR_LB,VAR_UB,self.Q));
                        k = cellfun(@(mu,var) GP.kernels.SG(mu,var, ...
                            [lowF,highF],[VAR_LB,VAR_UB]), ...
                            means, vars, 'un', 0);
                        kernels = GP.kernels.kernels(self.Q, k);
                        
                        % initialize coregionalization matrices
                        coregs = GP.coregs.mats(self.C,self.Q,modelOpts.R);
                        
                        % form GP with CSM kernel
                        self.LMCkernels{l} = GP.LMC_DFT(self.C,self.Q, ...
                            modelOpts.R,coregs,kernels,inf);
                        self.LMCkernels{l}.updateNoise = false;
                    end
                else
                    % intialize model from NMF
                    modelFreqs = self.freqBand(s);
                    power = abs(xFft(modelFreqs,:,:)).^2;
                    power = reshape(power, [], modelOpts.W);
                    [scoreInit, compInit] = nnmf(power', self.L);
                    
                    % normalize components and scores by max power in a channel
                    % for that component
                    compWeights = reshape(compInit', [], self.C, self.L);
                    compNorm = max(sum(compWeights, 1), [], 2);
                    compWeights = bsxfun(@rdivide, compWeights, compNorm);
                    compNorm = reshape(compNorm, 1, self.L);
                    scoreInit = bsxfun(@times, scoreInit, compNorm);
                    
                    % simulate overall power distribution by summing component
                    % weights to get relative importance of each frequency
                    freqWeights = round(sum(compWeights, 2)*DIST_PRECISION);
                    
                    for l = 1:self.L
                        simDist = randsample(s(modelFreqs), DIST_SAMPLES, true, freqWeights(:,1,l));
                        
                        % model component frequency usage via GMM
                        gmOpts = statset('MaxIter',GMM_MAX_ITER);
                        gmmodel = fitgmdist(simDist', self.Q,'Options',gmOpts,...
                            'RegularizationValue', 0.01);
                        means = num2cell( reshape(gmmodel.mu, 1, self.Q));
                        vars = num2cell( reshape(gmmodel.Sigma, 1, self.Q));
                        bWeight = gmmodel.ComponentProportion;
                        
                        k = cellfun(@(mu,var) GP.kernels.SG(mu,var, ...
                            [lowF,highF],[VAR_LB,VAR_UB]), ...
                            means, vars, 'un', 0);
                        kernels = GP.kernels.kernels(self.Q, k);
                        
                        % initialize coregionalization matrices
                        B = cell(self.Q,1);
                        for q = 1:self.Q
                            w = bWeight(q) * ones(self.C, modelOpts.R);
                            B{q} = GP.coregs.matComplex(self.C, modelOpts.R, w);
                        end
                        coregs = GP.coregs.mats(self.C,self.Q,modelOpts.R, B);
                        
                        % form GP with CSM kernel
                        self.LMCkernels{l} = GP.LMC_DFT(self.C,self.Q, ...
                            modelOpts.R,coregs,kernels,inf);
                        self.LMCkernels{l}.updateNoise = false;
                        
                        normConst = self.LMCkernels{l}.normalizeCovariance();
                        %scoreInit(:,l) = scoreInit(:,l)*normConst;
                    end
                    
                    self.scores = scoreInit';
                end
                
                self.updateKernels = true;
                self.updateScores = true;
                self.updateNoise = false;
            end
        end
        
        %{
    setPartitions: initializes variables needed to analyze model data in
        sequential partitions.
        %}
        function setPartitions(self, W, maxW)
            self.maxW = maxW;
            self.P = ceil(W/self.maxW);
            self.W = [];
            self.W(1:(self.P-1)) = self.maxW;
            self.W(self.P) = W - self.maxW*(self.P-1);
        end
        
        %{
    evaluate: Calculates the log likelihood for the current model on the
      dataset or a subset of the dataset.
      INPUTS
      s: frequencies corresponding to the first dimension of 'data'
      data: the full dataset on which the model has been trained. NxCxW
        array. N is the number of frequencies evaluated in the DFT when
        transforming the data to the frequency domain. C is the number of
        channels in the dataset. W is the number of windows.
      windowIdx: (optional) indicates the subset of windows for which the
        log likelihood is to be calculated. If unset, the log likelihood is
        calculated for all windows in the dataset.
        %}
        function LL = evaluate(self,s,data,windowIdx)
            nWindows = sum(self.W);
            if nargin < 4
                windowIdx = true(1,nWindows);
            end
            
            % restrict frequencies to those used by model
            modelFreqs = self.freqBand(s);
            s = s(modelFreqs);
            
            Ns = numel(s);
            Nc = self.C;
            Nl = self.L;
            
            % fill covariance matrix for each factor
            UKUlStore = zeros(Nc^2*Ns,Nl);
            for l = 1:Nl
                opts.block = true;
                opts.smallFlag = false;
                [~,UKUl] = self.LMCkernels{l}.UKU(s,opts);
                UKUlVals = self.LMCkernels{l}.extractBlocks(UKUl);
                UKUlStore(:,l) = UKUlVals(:);
            end
            
            [rows,cols] = find(logical(kron(speye(Ns),ones(Nc))));
            LL = 0;
            % loop through partitions
            for p = 1:self.P
                % get the desired scores and data from the current partition
                partitionIdx = ((p-1)*self.maxW+1):((p-1)*self.maxW+self.W(p));
                selectedWindows = false(1,nWindows);
                selectedWindows(partitionIdx) = windowIdx(partitionIdx);
                scoresAll = self.scores(:,selectedWindows);
                yP = data(modelFreqs,:,selectedWindows);
                
                % sum log likelihood for all windows in partition p
                for w = 1:sum(selectedWindows)
                    UKUvals = UKUlStore*scoresAll(:,w);
                    UKU = sparse(rows,cols,UKUvals,Nc*Ns,Nc*Ns);
                    noise = 1/self.eta*speye(Nc*Ns);
                    UKU = UKU + noise;
                    
                    % conversion to complex covariance matrix
                    UKU = 2*UKU;
                    
                    % vectorization of window w data
                    y = yP(:,:,w).';
                    y = y(:);
                    
                    logDetUKU = full(2*sum(log(diag(chol((UKU+UKU')./2)))));
                    LL = LL - Nc*Ns*log(pi) - logDetUKU - y'*(UKU\y);
                end
            end
            
            % remove machine precision complex values
            LL = real(LL);
        end
        
        %{
  The parameters that can be updated in the model are returned in a
  vector of the form [n;K;s]. Where n is a vector of parameters
  defining the variance of the white noise in the model. s is a
  vector of all factor scores. K is a vector of parameters defining
  the kernels of each factor and can be broken down into the
  parameter vectors for each kernel [K1 ... KL]. Each Kl can further
    be broken down into the vectors of parameters corresponding to the
    coregionalization matricies and the spectral gaussian components,
    (ie Kl = [Bl1 ... BlQ,kl1 ... klQ]). Blq corresponds to the qth
    coregionalization matrix of the lth factor. Each coregionalization
    matrix is rank R and complex (defined by R complex vectors b1...bR)
    The parameter vector Blq = [logA_b1 ... logA_bR;phi_b1 ... phi_bR]
    Where logA_bi represents a vector of C log magnitudes for the C
      elements of vector bi. Similarly, phi_bi is a vector of C-1 angles
      of bi relative to the 1st region. klq = [mu_q; var_q], the mean and
      variance of the spectral gaussian
        %}
        function res = getParams(self)
            if self.updateScores
                res = self.scores(:);
            else
                res = [];
            end
            if self.updateKernels
                params = cellfun(@(x)x.getParams,self.LMCkernels,'un',0);
                res = vertcat(params{:},res);
            end
            if self.updateNoise
                res = [self.eta; res];
            end
        end
        
        function setParams(self,params)
            [lb,ub] = self.getBounds();
            % constrain parameters
            params(params<lb) = lb(params<lb);
            params(params>ub) = ub(params>ub);
            
            indB = 1;
            if self.updateNoise
                self.eta = params(1); indB = indB +1;
            end
            if self.updateKernels
                for l = 1:self.L
                    indE = indB + self.LMCkernels{l}.nParams - 1; % get param indices
                    self.LMCkernels{l}.setParams(params(indB:indE));
                    indB = indE + 1;
                end
            end
            if self.updateScores
                self.scores = reshape(params(indB:end),[self.L,sum(self.W)]);
            end
        end
        
        function [lb,ub] = getBounds(self)
            if self.updateScores
                lb = zeros(self.L*sum(self.W),1);
                ub = 100*ones(self.L*sum(self.W),1);
            else
                lb = []; ub = [];
            end
            if self.updateKernels
                [lball,uball] = cellfun(@(x)x.getBounds,self.LMCkernels,'un',0);
                lb = vertcat(lball{:},lb);
                ub = vertcat(uball{:},ub);
            end
            if self.updateNoise
                lb = [1e-1; lb]; ub = [1e4; ub];
            end
        end
        
        % pIdx: structure containing boolean index vectors for each 'type' of
        % parameter for which a gradient is calculated
        %   FIELDS
        %   scores
        %   logParams
        %   sgMeans (if updateKernels)
        %   sgVars (if updateKernels)
        %   coregWeights (if updateKernels)
        %   coregShifts (if updateKernels)
        %   noise (if updateNoise)
        function pIdx = getParamIdx(self)
            nParams = numel(getParams(self));
            idxVec = false(1,nParams);
            if self.updateNoise
                pIdx.noise = idxVec; pIdx.noise(1) = 1;
                un = true;
            else
                un = false;
            end
            if self.updateKernels
                pIdx.sgMeans = idxVec; pIdx.sgVars = idxVec;
                pIdx.coregWeights = idxVec; pIdx.coregShifts = idxVec;
                
                nFactorParams = self.LMCkernels{1}.nParams;
                nKernelParams = self.LMCkernels{1}.coregs.B{1}.nParams;
                R = self.LMCkernels{1}.coregs.B{1}.R;
                
                for l = 1:self.L
                    idxl = nFactorParams*(l-1) + un;
                    for q = 1:self.Q
                        idxq = nKernelParams*(q-1);
                        wStart = idxq + idxl + 1;
                        wEnd = idxq + idxl + R*self.C;
                        sStart = wEnd + 1;
                        sEnd = idxq + idxl + nKernelParams;
                        pIdx.coregWeights(wStart:wEnd) = true;
                        pIdx.coregShifts(sStart:sEnd) = true;
                    end
                    mStart = sEnd + 1; vStart = mStart + 1;
                    vEnd = idxl + nFactorParams; mEnd = vEnd - 1;
                    pIdx.sgMeans(mStart:2:mEnd) = true;
                    pIdx.sgVars(vStart:2:vEnd) = true;
                end
            end
            if self.updateScores
                pIdx.scores = idxVec;
                nScores = numel(self.scores(:));
                pIdx.scores(end-nScores+1:end) = true;
            end
        end
        
        % gradient: returns gradients of the log-likelihood of the current
        %   model (given the data) with respect to the model parameters.
        % INPUTS:
        %   s: vector of frequency values corresponding to the first dimension
        %     of data
        %   data: NxCxW array of data in the frequency domain. N: number of
        %     freuquency domain points per window. C: number of channels in
        %     data. W: number of windows. If stochastic gradient descent is
        %     being used, only include windows for the current iteration.
        %   inds (optional): include if and only if stochastic gradient descent
        %     is being used. vector of indicies (integer values) of indicating
        %     which windows are included in the current iteration.
        % OUTPUTS:
        %   grad: vector of gradients for log likelihood with respect
        %     to all parameters
        %   maxCondNum: largest condition number of UKU for all windows
        %     evaluated
        function [grad, maxCondNum] = gradient(self,s,data,inds)
            %global gradCheck
            modelFreqs = self.freqBand(s);
            s = s(modelFreqs);
            Ns = numel(s); % number of frequency bins
            Nc = self.C; % number of channels
            Nl = self.L; % number of latent factors
            
            % initialize storage for derivatives and covariance matrices
            kderiv = cell(self.L,1);
            Bderiv = cell(self.L,1);
            UKUlStore = zeros(Nc^2*Ns,Nl);
            
            for l = 1:Nl
                % store Gram matrix for each factor (for speedup)
                opts.block = true;
                opts.smallFlag = false;
                [~,UKUl] = self.LMCkernels{l}.UKU(s,opts);
                vals = self.LMCkernels{l}.extractBlocks(UKUl);
                UKUlStore(:,l) = vals(:);
                
                % compute derivatives of parameters for each factor
                [kderiv{l},Bderiv{l}] = self.LMCkernels{l}.deriv(s);
            end
            kdAll = horzcat(kderiv{:}); % size Ns x L*nParams;
            BdAll = horzcat(Bderiv{:}); % size C^2 x L*nParams;
            
            % get indices of non-zero values in covariance matrices
            [rows,cols] = find(logical(kron(speye(Ns),ones(Nc))));
            
            %initialize gradient storage
            nParams = self.LMCkernels{1}.nParams;
            LMCgrad = zeros(nParams,self.L);
            ngrad = 0;
            sgrad = zeros(self.L,sum(self.W));
            
            % loop through all memory partitions (unless using sgd)
            if nargin == 3
                Parts = self.P;
                stochastic = false;
            elseif nargin == 4
                Parts = 1;
                stochastic = true;
            end
            for p = 1:Parts
                
                % extract data and factors scores for partition p
                if stochastic
                    y = data(modelFreqs,:,:);
                else
                    inds = ((p-1)*self.maxW+1):((p-1)*self.maxW+self.W(p));
                    y = data(modelFreqs,:,inds);
                end
                y = conj(y);
                theseScores = self.scores(:,inds);
                
                % initialize condition number tracking variable
                if nargout >=2, maxCondNum = 0; end
                
                % obtain inverse of Gram matrix for each window in the
                % partition.
                for w = 1:numel(inds)
                    % get UKU for window w
                    vals = UKUlStore*theseScores(:,w);
                    UKU = sparse(rows,cols,vals,Nc*Ns,Nc*Ns);
                    noise = 1/self.eta*speye(Nc*Ns);
                    UKU = UKU + noise;
                    
                    % get inverse UKU
                    LHS = repmat(eye(Nc),[1,Ns]);
                    UKUi = LHS / UKU;
                    UKUinv = reshape(UKUi, [Nc Nc Ns]);
                    
                    if nargout >=2
                        % calculate L1 condition number
                        thisCondNum = norm(UKU,1)*norm(UKUi,1);
                        maxCondNum = max(thisCondNum,maxCondNum);
                    end
                    
                    % get A for gradient computations (see Williams & Rasmussen)
                    Ag = self.getA(y(:,:,w),UKUinv);
                    Agt = Ag';
                    
                    if self.updateKernels
                        % gradient for window
                        gradW = (Agt*BdAll).*kdAll;
                        gradW = sum(real(gradW),1).'; % i.e. trace
                        thisKgrad = (s(2)-s(1)) * Ns * bsxfun(@times, theseScores(:,w)', ...
                            reshape(gradW,[nParams,Nl]) ); % check this!
                        
                        % add window w contribution to gradient
                        LMCgrad = LMCgrad + thisKgrad;
                        %if strcmp(gradCheck,'kernels')
                        %  util.gradientCheckKernels
                        %end
                    end
                    
                    if self.updateNoise
                        % gradient for window w noise = tr(Ag) * -eta^-2
                        traceInd = eye(Nc,'logical');
                        traceInd = repmat(traceInd(:),[1,Ns]);
                        thisNgrad = - 1/(self.eta^2)*sum(real(Ag(traceInd)));
                        ngrad = ngrad + thisNgrad;
                        %                 if strcmp(gradCheck,'noise')
                        %                   util.gradientCheckNoise
                        %                 end
                    end
                    if self.updateScores
                        % gradient for window w factor scores
                        sgrad(:,inds(w)) = real(Ag(:)'*UKUlStore)';
                        %if strcmp(gradCheck,'scores')
                        %  util.gradientCheckScores
                        %end
                    end
                end
            end
            
            %if strcmp(gradCheck,'coregs')
            %  util.gradientCheckCoregs
            %end
            
            % concatenate gradients if necessary
            if self.updateScores
                grad = sgrad(:);
            else
                grad = [];
            end
            if self.updateKernels
                grad = vertcat(LMCgrad(:),grad); 
            end
            if self.updateNoise
                grad = vertcat(ngrad, grad);
            end
        end
        
        % normalize factors for identifiability
        function makeIdentifiable(self)
            for l = 1:self.L
                normConst = self.LMCkernels{l}.normalizeCovariance();
                self.scores(l,:) = self.scores(l,:)*normConst;
            end
        end
        
        function res = UKU(self,s,n,UKUlstore)
            s = s(self.freqBand(s));
            Ns = numel(s);
            
            res = bsxfun(@times,1/self.eta*eye(self.C),ones([1,1,Ns]));
            if ~exist('UKUlstore','var')
                for l = 1:self.L
                    res = res + self.scores(l,n) * ...
                        self.LMCkernels{l}.extractBlocks(self.UKUl(s,l));
                end
            else
                res = res + sum(bsxfun(@times,UKUlstore, ...
                    permute(self.scores(:,n),[2,3,4,1])),4);
            end
        end
        
        function res = UKUl(self,s,l)
            Nc = self.C;
            Ns = numel(s);
            d = 1/(2*Ns*(s(2)-s(1)));
            SD = self.LMCkernels{l}.kernels.specDens(s);
            
            res = spalloc(Ns*Nc,Ns*Nc,Nc^2*Ns);
            for q = 1:self.Q
                B = self.LMCkernels{l}.coregs.getMat(q);
                res = res + 1/d/2*kron(spdiags(SD(:,q),0,Ns,Ns),B);
            end
        end
        
        % draw random timeseries from model
        % INPUTS
        %  N: points per sampled window
        %  delta: sampling period
        %  scores (optional): scores for sampled data
        function [y,scores,UKU] = sample(self,N,delta,scores)
            % set frequencies to evaluate
            Ns = ceil(N/2);
            s = 1/(N*delta):1/(N*delta):1/(2*delta);
            
            % generate scores if necessary
            if nargin < 4
                scores = zeros(1,self.L);
                nActive = randi([2 self.L-1]);
                activeIdx = randsample(self.L,nActive);
                scores(activeIdx) = rand(1,nActive);
                scores = scores/sqrt(sum(scores.^2)); %approx unit variance
            end
            
            % generate covariance matrix
            UKU = bsxfun(@times,1/self.eta*eye(self.C),ones([1,1,Ns]));
            for l = 1:self.L
                UKU = UKU + scores(l) * ...
                    self.LMCkernels{l}.extractBlocks(self.UKUl(s,l));
            end
            
            Uy = zeros(Ns,self.C);
            for n = 1:Ns
                UKUn = UKU(:,:,n);
                UKUnri = 2*[real(UKUn) imag(-UKUn); imag(UKUn) real(UKUn)];
                Uyri = mvnrnd(zeros(2*self.C,1),UKUnri);
                Uy(n,:) = Uyri(1:self.C) + 1i*Uyri(self.C+1:end);
            end
            y = real(ifft([Uy; flipud(Uy)]));
        end
        
        function plotCsd(self,n,varargin)
            % plot the cross-spectral density for window n
            p = inputParser;
            p.KeepUnmatched = true;
            addOptional(p,'minFreq',0,@isnumeric);
            addOptional(p,'maxFreq',30,@isnumeric);
            parse(p,varargin{:});
            opts = p.Results;
            
            s = linspace(opts.minFreq,opts.maxFreq,1e3);
            s = s(self.freqBand(s));
            UKU = self.UKU(s,n);
            
            plotCsd@GP.spectrumPlots(s,UKU,varargin{:})
        end
        
        function keepChannels(self,chans)
            % replace all properties with only the channels in chans
            % NOTE: this will remove data!
            self.C = numel(chans);
            for l = 1:self.L
                self.LMCkernels{l}.keepChannels(chans);
            end
        end
        
        function plotRelativeCsd(self,l,varargin)
            % plot the spectral density normalized accross all factors.
            % This gives the *relative* impact of each factor as a function
            % of frequency.
            p = inputParser;
            p.KeepUnmatched = true;
            addOptional(p,'minFreq',0,@isnumeric);
            addOptional(p,'maxFreq',30,@isnumeric);
            parse(p,varargin{:});
            opts = p.Results;
            
            s = linspace(opts.minFreq,opts.maxFreq,1e3);
            
            UKU = zeros(self.C,self.C,1e3,self.L);
            optsUKU.smallFlag = true;
            optsUKU.block = true;
            for j = 1:self.L
                UKU(:,:,:,j) = self.LMCkernels{j}.UKU(s,optsUKU);
            end
            UKUnorm = bsxfun(@rdivide,UKU,sum(abs(UKU),4));
            
            GP.spectrumPlots.plotCsd(s,UKUnorm(:,:,:,l),varargin{:})
        end
        
        % get frequencies modeled by this object
        function fBand = freqBand(self,s)
            fBand = s >= self.freqBounds(1) & s <= self.freqBounds(2);
        end
        
        % Make a copy of a handle object.
        function new = copy(self)
            % Instantiate new object of the same class.
            new = feval(class(self));
            
            % Copy all non-hidden properties.
            p = properties(self);
            for i = 1:length(p)
                % check if property is actually a cell array of properties
                if  iscell(self.(p{i}))
                    % instantiate new cell array
                    new.(p{i}) = cell(size(self.(p{i})));
                    for j = 1:numel(self.(p{i}))
                        % if property is a handle, deep copy
                        if ismethod(self.(p{i}){j},'copy')
                            new.(p{i}){j} = self.(p{i}){j}.copy;
                        else
                            new.(p{i}){j} = self.(p{i}){j};
                        end
                    end
                else
                    % if property is a handle, deep copy
                    if ismethod(self.(p{i}),'copy')
                        new.(p{i}) = self.(p{i}).copy;
                    else
                        new.(p{i}) = self.(p{i});
                    end
                end
            end
        end
        
    end
    
    methods(Static)
        function Avec = getA(data,UKUinv)
            % get matrix A = a*a' - W*K^{-1} for GP gradient steps
            % dL/dp = tr(A*dK/dp)
            dataRow = permute(data,[3,2,1]);
            dataCol = permute(conj(data),[2,3,1]);
            alpha1 = sum(bsxfun(@times,UKUinv,dataRow),2);
            alpha2 = sum(bsxfun(@times,UKUinv,dataCol),1);
            Akeep = (1/2)*bsxfun(@times,alpha1,alpha2) - UKUinv;
            
            [Ns,Nc] = size(data);
            Avec = reshape(Akeep,[Nc^2,Ns]); % vectorize
        end
        
        % draw a random vector from uniform distribution on the logarithmic space
        % between two numbers
        function x = logRand(lb,ub,n)
            logLb = log(lb); logUb = log(ub);
            diff = logUb - logLb;
            logX = logLb + diff*rand(1,n);
            x = exp(logX);
        end
    end
end
