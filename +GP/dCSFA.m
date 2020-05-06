classdef dCSFA < handle
    properties
        W % number of time windows (observations) per animal (A-dim vector)
        C % number of channels
        Q % number of components in SM kernel
        L % number of factors in mixture
        dIdx % indicies of predictive factors
        lambda % lagrange multiplier (SVM tuning parameter)
        
        kernel % CSFA kernel
        
        labels % labels for training classifier
        classModel % classifier
        classType % type of classifier (svm or logistic)
        group % groups for mixed-intercept models
    end
    
    methods
        function self = dCSFA(modelOpts,labels,group,s,xFft)
            if nargin > 0
                self.W = int64(modelOpts.W);
                self.C = modelOpts.C;
                self.Q = modelOpts.Q;
                self.L = int64(modelOpts.L);

                self.lambda = modelOpts.lambda;
                
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
                
                % set supervised factors; if dIdx not set, assume 1 supervised factor
                if ~isfield(modelOpts,'dIdx'); modelOpts.dIdx = 1; end
                if length(modelOpts.dIdx) > 1
                    % treat dIdx as boolean vector
                    if numel(modelOpts.dIdx) ~= modelOpts.L
                        error(['vector identifying discriminitive factors'...
                            '(modelOpts.dIdx) does not match total number of factors'])
                    end
                    self.dIdx = modelOpts.dIdx;
                else
                    % treat dIdx as scalar giving number of supervised factors
                    self.dIdx = util.selectDiscFactors(modelOpts.dIdx,labels,...
                        self.kernel.scores);
                end
                
                % check for binary labels
                labelList = unique(labels);
                if length(labelList) == 2
                    if islogical(labels)
                        self.labels = labels;
                    else
                        % convert to boolean
                        self.labels = labels == labelList(2);
                    end
                else
                    % if not binary, convert to one-hot
                    self.labels = util.oneHot(labels);
                end
                
                self.classType = modelOpts.discrimModel;
                if nargin >= 3 && ~isempty(group)
                    self.group = group;
                    if strcmp(self.classType,'multinomial')
                        error(['Cannot use mixed intercept model with a multinomial ' ...
                            'classifier.'])
                    end
                else
                    % all windows are same group
                    self.group = ones(1,modelOpts.W);
                end
            end
        end
        
        function res = evaluate(self,s,dat)
            % evaluate objective function
            y = self.labels;
            features = self.getFeatures();
            
            % calculated classifier loss
            switch self.classType
                case 'svm'
                    % remove constant term if not a mixed model
                    if numel(unique(self.group)) == 1
                        features = features(1:end-1,:);
                    end
                    
                    [~,yHat] = predict(self.classModel,features');
                    % convert boolean labels to +1/-1
                    y = y*2 - 1;
                    loss = sum(max(0,1-y.*yHat(:,1)));
                    % hinge loss
                case 'logistic'
                    % assumes boolean labels
                    yHat = self.classModel.Fitted.Probability;
                    loss = sum(-y.*log(yHat+eps) + (1-y).*log(1-yHat+eps));
                    % cross-entropy
                case 'multinomial'
                    yHat = mnrval(self.classModel,features(1:end-1,:)');
                    loss = sum(sum(-y.*log(yHat'+eps)));
                    % cross-entropy
            end
            
            res = self.kernel.evaluate(s,dat) - self.lambda*loss;
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
                features = self.getFeatures;
                
                % choose gradient method for appropriate classifier
                if isa(self.classType,'function_handle')
                    gradClass = zeros(self.L,self.W);
                    [self.classModel, gc] = self.classType(self.labels,features);
                    gradClass(self.dIdx,:) = gc;
                else
                    switch self.classType
                        case 'svm'
                            gradClass = self.svmGradient(self.labels,features);
                        case 'logistic'
                            gradClass = self.logisticGradient(self.labels,features);
                        case 'multinomial'
                            gradClass = self.multiGradient(self.labels,features);
                    end
                end
                
                % if using stochastic learning, only update scores in current batch
                if nargin >=4
                    indsMask = false(1,self.W);
                    indsMask(inds) = true;
                    gradClass(:,~indsMask) = 0;
                end
                
                grad(end-self.L*self.W+1:end) = grad(end-self.L*self.W+1:end) - ...
                    self.lambda*reshape(gradClass,[self.L*self.W,1]);
            end
        end
        
        function makeIdentifiable(self)
            self.kernel.makeIdentifiable;
        end
        
        function grad = svmGradient(self,thisLabel,features)
            % remove constant term if not a mixed model
            if numel(unique(self.group)) == 1
                features = features(1:end-1,:);
            end
            
            % 1) convert boolean labels to +1/-1
            y = thisLabel*2 - 1;
            
            % 2) obtain the SVM weights on classifying factor scores
            cmodel = fitcsvm(features', y, 'KernelFunction', 'linear', 'Prior','uniform',...
                'CacheSize','maximal');
            self.classModel = cmodel; % store classification model
            
            % 3) compute gradient of this hinge loss function
            % find support vectors
            sv = cmodel.IsSupportVector;
            % take gradient of hinge loss function
            gradHL = sv .* (-y*cmodel.Beta');
            
            % 4) inject gradient of hinge loss into kernel gradient
            grad = zeros(self.L,self.W);
            grad(self.dIdx,:) = gradHL;
        end
        
        
        function grad = logisticGradient(self,thisLabel,features)
            self.classModel = fitglm(features',thisLabel,'Distribution','binomial',...
                'Intercept',false);
            b = self.classModel.Coefficients.Estimate;
            nDFactors = sum(self.dIdx);
            coeffs = b(1:nDFactors);
            
            % 3) gradient of cross-entropy loss
            yHat = self.classModel.Fitted.Probability;
            gradCE = -coeffs*(thisLabel - yHat)';
            
            % 4) inject gradient of hinge loss into kernel gradient
            grad = zeros(self.L,self.W);
            grad(self.dIdx,:) = gradCE;
        end
        
        function grad = multiGradient(self,thisLabel,features)
            features = features(1:end-1,:);
            self.classModel = mnrfit(features',thisLabel');
            
            % 3) gradient of cross-entropy loss
            yHat = mnrval(self.classModel,features');
            gradCE = -self.classModel(2:end,:)*(thisLabel(1:end-1,:) - yHat(:,1:end-1)');
            
            % 4) inject gradient of hinge loss into kernel gradient
            grad = zeros(self.L,self.W);
            grad(self.dIdx,:) = gradCE;
        end
        
        function features = getFeatures(self)
            % get features for classifier
            scores = self.kernel.scores; 
            scores = scores(self.dIdx,:); % (gradients taken wrt log scores)
            features = [scores' util.oneHot(self.group)']';
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
