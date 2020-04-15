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
        function self = dCSFA(modelOpts,labels,group)
            if nargin > 0
                self.W = int64(modelOpts.W);
                self.C = modelOpts.C;
                self.Q = modelOpts.Q;
                self.L = int64(modelOpts.L);
                if numel(modelOpts.dIdx) ~= modelOpts.L
                    error(['Specified number of discriminitive factors is greater than'...
                        ' total number of factors'])
                end
                self.dIdx = logical(modelOpts.dIdx);
                self.lambda = modelOpts.lambda;
                
                if isfield(modelOpts,'kernel')
                    self.kernel = modelOpts.kernel;
                else
                    self.kernel = GP.CSFA(modelOpts);
                end
                
                if islogical(labels)
                    self.labels = labels;
                else
                    self.labels = util.oneHot(labels);
                end
                
                self.classType = modelOpts.discrimModel;
                if nargin >= 3
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
                    [y,features] = self.balanceClasses(y,features);
                    [~,yHat] = predict(self.classModel,features');
                    % convert boolean labels to +1/-1
                    y = y*2 - 1;
                    loss = sum(max(0,1-y.*yHat(:,1)));
                    % hinge loss
                case 'logistic'
                    % assumes boolean labels
                    yHat = self.classModel.Fitted.Probability;
                    loss = sum(-y.*log(yHat) + (1-y).*log(1-yHat));
                    % cross-entropy
                case 'multinomial'
                    yHat = mnrval(self.classModel,features');
                    loss = sum(sum(-y.*log(yHat')));
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
        
        function makeIdentifiable(self)
            self.kernel.makeIdentifiable;
        end
        
        function grad = svmGradient(self,thisLabel,features)
            % Get gradient of classifier loss w/ respect to scores
            [labelsSVM,scoresSVM] = self.balanceClasses(thisLabel,features);
            
            % 2) obtain the SVM weights on classifying factor scores
            cmodel = fitcsvm(scoresSVM',labelsSVM);
            self.classModel = cmodel; % store classification model
            
            % 3) compute gradient of this hinge loss function
            lossInd = (1 - (thisLabel .* (features'*abs(cmodel.Beta) + cmodel.Bias))) > 0; % find support vectors
            gradHL = -1 * bsxfun(@times, (lossInd .* thisLabel)',abs(cmodel.Beta)) .* features; % take gradient of hinge loss function
            
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
            params = self.kernel.getParams; % obtain current parameter value vector
            % extract the log of factor scores from parameter values
            scores = reshape(params(end-self.L*self.W+1:end),[self.L, ...
                self.W]);
            scores = scores(self.dIdx,:); % (gradients taken wrt log scores)
            features = [scores' util.oneHot(self.group)']';
        end
        
        function [labelsBal, featuresBal] = balanceClasses(self,labels,features)
            % if more negative examples than positive example,
            % upsample positive examples, or vice versa
            if sum(labels==1) < sum(labels==0)
                [~,idx] = datasample(features,sum(labels==0),2,'Weights',double(labels));
                labelsBal = [labels(idx); labels(labels==0)];
                featuresBal = [features(:,idx), features(:,labels==0)];
            else % if more positive examples than negative
                % examples, upsample negative examples
                [~,idx] = datasample(features,sum(labels==1),2,'Weights',double(labels==0));
                labelsBal = [labels(labels==1); labels(idx)];
                featuresBal = [features(:,labels==1), features(:,idx)];
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
