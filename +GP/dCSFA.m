classdef dCSFA < handle
    properties
        W % number of time windows (observations) per animal (A-dim vector)
        C % number of channels
        Q % number of components in SM kernel
        L % number of factors in mixture
        kernel % CSFA kernel        
        
        dIdx % indicies of predictive factors
        S % number of supervised sub-models
        lambda % supervision strength
        labels % labels for training classifier
        classModel % classifier(s)
        classType % type of classifier (svm or logistic)
        group % groups for mixed-intercept models
        isMixed % indicates if each supervised model has a mixed intercept
        scoreNorm % normalization applied to scores before input to classifier
        isWindowSupervised % indicates which windows to supervise
        classWeights % observation weights for classifiers
    end
    
    methods
        function self = dCSFA(modelOpts,labels,group,s,xFft)
            if nargin > 0
                self.W = int64(modelOpts.W);
                self.C = modelOpts.C;
                self.Q = modelOpts.Q;
                self.L = int64(modelOpts.L);

                S = size(labels,2);
                if isa(labels, 'cell') && (S>1 || size(labels,1)==1)
                    self.encodeLabels(labels)
                    self.S = S;
                else
                % if multiple sets of labels aren't provided
                % pass as single cell entry
                    self.encodeLabels({labels})
                    self.S = 1;
                end
                
                % duplicate lambda if necessary
                if self.S>1 && numel(modelOpts.lambda)==1
                    self.lambda = repmat(modelOpts.lambda,1,self.S);
                else
                    self.lambda = modelOpts.lambda;
                end
                
                % if isWindowSupervised provided, check if duplication necessary
                if isfield(modelOpts, 'isWindowSupervised')
                    if self.S>1 && size(modelOpts.isWindowSupervised,2)==1
                        self.isWindowSupervised = repmat(modelOpts.isWindowSupervised,1,self.S);
                    else
                        self.isWindowSupervised = modelOpts.isWindowSupervised;
                    end
                else % set all windows to supervised
                    self.isWindowSupervised = ones(self.W, self.S);
                end
                
                if isfield(modelOpts,'kernel')
                    self.kernel = modelOpts.kernel;
                else
                    if nargin < 4
                        % initialize CSFA kernel randomly
                        self.kernel = GP.CSFA(modelOpts);
                    else
                        % initialize using NMF on xFft
                        self.kernel = GP.CSFA(modelOpts,s,xFft);
                    end
                end
                
                if isfield(modelOpts,'classWeights')
                   self.classWeights = modelOpts.classWeights;
                else
                    self.classWeights = cell(1,S);
                    for s = 1:S
                        self.classWeights{s} = ones(sum(self.isWindowSupervised(:,s)),1);
                    end
                end
                
                % set supervised factors
                if length(modelOpts.dIdx) > 1
                    % treat dIdx as boolean vector
                    if numel(modelOpts.dIdx) ~= modelOpts.L
                        error(['vector identifying discriminitive factors'...
                            '(modelOpts.dIdx) does not match total number of factors'])
                    end
                    self.dIdx = modelOpts.dIdx;
                else
                    % only use non-adversarial supervised models for picking supervised factors
                    nonAdv = self.lambda>0;
                    self.dIdx = util.selectDiscFactors(modelOpts.dIdx, labels(nonAdv),...
                        self.kernel.scores, self.isWindowSupervised(:,nonAdv));
                end

                if numel(modelOpts.discrimModel)~=self.S
                    error(['Must have an entry in modelOpts.discrimModel for each '...
                        'discriminative task'])
                end
                self.classType = modelOpts.discrimModel;
                
                if nargin >= 3 && ~isempty(group)
                    if S>1 && size(group,2)==1
                        self.group = repmat(group,1,self.S);
                    elseif isa(group,'cell')
                        self.group = group;
                    else
                        error('group parameter must be a cell array')
                    end
                    
                    for s = 1:S
                        self.isMixed(s) = numel(unique(self.group{s})) > 1;
                    end
                else
                    % all windows are same group
                    self.group = repmat({ones(1,modelOpts.W)}, 1, self.S);
                    self.isMixed = false(1,self.S);
                end
            end
        end
        
        function encodeLabels(self, labelTypes)
            for l = 1:length(labelTypes)
                % check for binary labels
                labelList = unique(labelTypes{l});
                if length(labelList) == 2
                    if ~islogical(labelTypes{l})
                        % convert to boolean
                        labelTypes{l} = labelTypes{l} == labelList(2);
                    end
                else
                    % if not binary, convert to one-hot
                    labelTypes{l} = util.oneHot(labelTypes{l})';
                end
            end
            
            self.labels = labelTypes;
        end
        
        function [ll, cLoss] = evaluate(self,s,dat)
            % evaluate objective function
            yList = self.labels;
                        
            % get cumulative sum of all classifier losses
            K = length(yList);
            cLoss = zeros(1,K);
            for k = 1:K
                y = yList{k}(self.isWindowSupervised(:,k));
                features = self.getFeatures(k);

                % calculate classifier loss
                switch self.classType{k}
                    case 'svm'
                        % convert boolean labels to +1/-1
                        y = y*2 - 1;
                        % hinge loss
                        m = margin(self.classModel{k}, features', y)/2;
                        hl = max(0,1-m);
                        cLoss(k) = sum(hl);
                        
                    case 'logistic'
                        % assumes boolean labels
                        [~,yHat] = predict(self.classModel{k}, features');
                        trueIdx = self.classModel{k}.ClassNames;
                        yHat = yHat(:,trueIdx);
                        % cross-entropy
                        ce = -y.*log(yHat+eps) - (1-y).*log(1-yHat+eps);
                        cLoss(k) = sum(ce);
                        
                    case 'multinomial'
                        yHat = mnrval(self.classModel{k},features');
                        % cross-entropy
                        ce = sum(-y'.*log(yHat'+eps));
                        cLoss(k) = sum(ce);
                end
            end
            
            ll = self.kernel.evaluate(s,dat);
        end
        
        function res = getParams(self)
            res = self.kernel.getParams;
        end
        
        function res = eta(self)
            res = self.kernel.eta;
        end
        
        function res = scores(self)
            res = self.kernel.scores;
        end
        
        function setParams(self,params)
            self.kernel.setParams(params);
        end
        
        function [lb,ub] = getBounds(self)
            [lb,ub] = self.kernel.getBounds;
        end
        
        function update = updateKernels(self)
            update = self.kernel.updateKernels;
        end
        
        function update = updateScores(self)
            update = self.kernel.updateScores;
        end
        
        function setUpdateState(self, updateKernels, updateScores)
           self.kernel.updateKernels = updateKernels;
           self.kernel.updateScores = updateScores;
        end
        
        % pIdx: structure containing boolean index vectors for each 'type' of
        % parameter for which a gradient is calculated
        %   FIELDS
        %   scores
        %   sgMeans (if updateKernels)
        %   sgVars (if updateKernels)
        %   coregWeights (if updateKernels)
        %   coregShifts (if updateKernels)
        %   noise (if updateNoise)
        function pIdx = getParamIdx(self)
            pIdx = self.kernel.getParamIdx;
        end
        
        function [grad, condNum] = gradient(self,s,data,inds)
            % 1) obtain gradient and current kernel parameter values
            if nargin <=3
                [grad, condNum] = self.kernel.gradient(s,data);
            else % use stochastic learning
                [grad, condNum] = self.kernel.gradient(s,data,inds);
            end

            if self.updateScores
                % iterate through each set of supervised labels and add gradients
                gradClass = zeros(self.L,self.W);
                for k = 1:length(self.labels)
                    thisLabels = self.labels{k}(self.isWindowSupervised(:,k),:);
                    features = self.getFeatures(k);
                
                    % choose gradient method for appropriate classifier
                    if isa(self.classType{k},'function_handle')
                        thisGrad = zeros(self.L,self.W);
                        [self.classModel{k}, gc] = self.classType{k}(thisLabels,features);
                        thisGrad(self.dIdx,:) = gc;
                    else
                        switch self.classType{k}
                            case 'svm'
                                thisGrad = self.svmGradient(thisLabels,features,k);
                            case 'logistic'
                                thisGrad = self.logisticGradient(thisLabels,features,k);
                            case 'multinomial'
                                thisGrad = self.multiGradient(thisLabels,features,k);
                        end
                    end
                    
                    gradClass = gradClass + self.lambda(k)*thisGrad;
                end
                
                % if using stochastic learning, only update scores in current batch
                if nargin >=4
                    indsMask = false(1,self.W);
                    indsMask(inds) = true;
                    gradClass(:,~indsMask) = 0;
                end
                
                scoreIdx = self.getParamIdx.scores;
                grad(scoreIdx) = grad(scoreIdx) - reshape(gradClass,[self.L*self.W,1]);
            end
        end
        
        function makeIdentifiable(self)
            self.kernel.makeIdentifiable;
        end
        
        function grad = svmGradient(self, thisLabel, features, sIdx)
            %global gradCheck
            
            % 1) convert boolean labels to +1/-1
            y = thisLabel*2 - 1;
            
            % 2) obtain the SVM weights on classifying factor scores
            cmodel = fitcsvm(features', y, 'KernelFunction','linear', 'Prior','uniform',...
                'CacheSize','maximal', 'Weights',self.classWeights{sIdx});
            self.classModel{sIdx} = cmodel; % store classification model
            
            % 3) compute gradient of this hinge loss function
            % find support vectors
            sv = cmodel.IsSupportVector;
            % take gradient of hinge loss function
            nDFactors = sum(self.dIdx);
            normBeta = cmodel.Beta(1:nDFactors) ./ self.scoreNorm;
            gradHL = sv .* (-y*normBeta');
            
            % 4) inject gradient of hinge loss into kernel gradient
            grad = zeros(self.L,self.W);
            grad(self.dIdx, self.isWindowSupervised(:,sIdx)) = gradHL';
            
%             if strcmp(gradCheck,'svm')
%                 util.gradientCheckSVM
%             end
        end
        
        
        function grad = logisticGradient(self, thisLabel, features, sIdx)
            %global gradCheck
            self.classModel{sIdx} = fitclinear(features', thisLabel, 'Prior','uniform',...
                'Learner','logistic', 'Solver','lbfgs', 'FitBias',~self.isMixed(sIdx),...
                'Weights', self.classWeights{sIdx});
            b = self.classModel{sIdx}.Beta;
            
            % remove coefficients for non-score features
            nDFactors = sum(self.dIdx);
            normCoeffs = b(1:nDFactors)./self.scoreNorm;
            
            % 3) gradient of cross-entropy loss
            [~,yHat] = predict(self.classModel{sIdx}, features');
            trueIdx = self.classModel{sIdx}.ClassNames;
            gradCE = -normCoeffs*(thisLabel - yHat(:,trueIdx))';
            
            % 4) inject gradient of hinge loss into kernel gradient
            grad = zeros(self.L,self.W);
            grad(self.dIdx, self.isWindowSupervised(:,sIdx)) = gradCE;
            
%             if strcmp(gradCheck,'logistic')
%                 util.gradientCheckLogistic
%             end
        end
        
        function grad = multiGradient(self, thisLabel, features, sIdx)
            % remove labels with no representation
            noRep = sum(thisLabel,1) == 0;
            thisLabel(:,noRep) = [];
            
            self.classModel{sIdx} = mnrfit(features',thisLabel);
            
            % 3) gradient of cross-entropy loss
            yHat = mnrval(self.classModel{sIdx},features');
            nDFactors = sum(self.dIdx);
            normCoeffs = self.classModel{sIdx}(2:nDFactors+1,:)./self.scoreNorm;
            gradCE = -normCoeffs*(thisLabel(:,1:end-1)' - yHat(:,1:end-1)');
            
            % 4) inject gradient of hinge loss into kernel gradient
            grad = zeros(self.L,self.W);
            grad(self.dIdx, self.isWindowSupervised(:,sIdx)) = gradCE;
        end
        
        function features = getFeatures(self, sIdx)
            % get features for classifier
            scores = self.kernel.scores;
            wSupervised = self.isWindowSupervised(:,sIdx);
            scores = scores(self.dIdx, wSupervised);
            
            %normalize scores by rms values
            self.scoreNorm = sqrt(mean(scores.^2, 2));
            
            features = bsxfun(@rdivide, scores, self.scoreNorm);
            if self.isMixed(sIdx)
                features = [features; util.oneHot(self.group{sIdx}(wSupervised))];
            end
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
