classdef matComplex < handle
    % A complex coregionalization matrix
    
    properties
        % model parameters
        C % number of channels
        R % rank of coregionalization matrix
        
        logWeights % real component of coregionalization matrix
        shifts % imaginary component of coregionalization matrix
        
        weightsB % upper/lower bounds on weights
        shiftsB % upper/lower bounds on shifts
    end
    
    methods
        function self = matComplex(C,R,weights,shifts,weightsB,shiftsB)
            LOGW_LB = -9.9; LOGW_UB = 3;
            if nargin > 0
                self.C = C;
                self.R = R;
                
                if ~exist('weights','var') || isempty(weights)
                    self.logWeights = log(1/10*rand(C,R)+exp(LOGW_LB));%gamrnd(3,1,[C,R]);
                else
                    self.logWeights = log(weights);
                end
                if ~exist('shifts','var') || isempty(shifts)
                    self.shifts = pi/2 * randn(C,R); %2*pi*rand(C,R);
                    self.shifts(1,:) = 0;
                else
                    self.shifts = shifts;
                end
                if ~exist('weightsB','var') || isempty(weightsB)
                    self.weightsB = cat(3,LOGW_LB*ones(C,R),LOGW_UB*ones(C,R));
                else
                    self.weightsB = weightsB;
                end
                
                if ~exist('shiftsB','var') || isempty(shiftsB)
                    self.shiftsB = cat(3,-inf(C,R),inf(C,R));
                else
                    self.shiftsB = shiftsB;
                end
            end
        end
        
        function res = getMat(self)
            w = exp(self.logWeights);
            psi = self.shifts;
            
            b = (w .* exp(-1i*psi));
            res = b*b'; % this is the complex conjugate
        end
        
        function res = getParams(self)
            psi = self.shifts(2:end,:);
            res = [self.logWeights(:); wrapToPi(psi(:))];
        end
        
        function setParams(self,params)
            self.logWeights = reshape(params(1:self.C*self.R),[self.C,self.R]);
            self.shifts(2:end,:) = ...
                reshape(params(self.C*self.R+1:end),[self.C-1,self.R]);
            %self.shifts = zeros(self.C,self.R);
        end
        
        function [lb,ub] = getBounds(self)
            lbw = self.weightsB(:,:,1);
            lbs = self.shiftsB(2:end,:,1);
            ubw = self.weightsB(:,:,2);
            ubs = self.shiftsB(2:end,:,2);
            
            lb = [lbw(:);lbs(:)];
            ub = [ubw(:);ubs(:)];
        end
        
        function res = deriv(self)
            %             cvals = repmat(1:self.C,[1,self.R])';
            %             rvals = kron(1:self.R,ones(1,self.C))';
            %
            %             cvals2 = repmat(2:self.C,[1,self.R])';
            %             rvals2 = kron(1:self.R,ones(1,self.C-1))';
            %
            %             res = [arrayfun(@(c,r)self.derivWeight(c,r),cvals,rvals,'un',0); ...
            %                 arrayfun(@(c,r)self.derivShift(c,r),cvals2,rvals2,'un',0)];
            
            cvals = repmat(1:self.C,[1,self.R])';
            rvals = kron(1:self.R,ones(1,self.C))';
            
            w = exp(self.logWeights);
            p = exp(-1i*self.shifts);
            dp = -1i*p;
            
            res1 = cell(size(cvals));
            res2 = cell([numel(cvals)-self.R,1]);
            res1(:) = {zeros(self.C)};
            res2(:) = {zeros(self.C)};
            
            for i = 1:numel(cvals)
                c = cvals(i); r = rvals(i);
                
                res1{i}(:,c) = res1{i}(:,c) + w(:,r)*w(c,r);
                res1{i}(c,:) = res1{i}(c,:) + w(:,r)'*w(c,r);
                res1{i} = res1{i} .* (p(:,r)*p(:,r)');
                
                if c > 1
                    res2{i-r}(:,c) = res2{i-r}(:,c) + p(:,r)*conj(dp(c,r));
                    res2{i-r}(c,:) = res2{i-r}(c,:) + p(:,r)'*dp(c,r);
                    res2{i-r} = res2{i-r} .* (w(:,r)*w(:,r)');
                end
            end
            
            res = vertcat(res1,res2);
        end
        
        function res = derivWeight(self,c,r)
            w = self.weights(:,r);
            psi = self.shifts(:,r);
            
            vc = zeros(self.C,1); vc(c) = 1;
            p = exp(-1i*psi);
            
            res = (w*vc' + vc*w') .* (p*p');
        end
        
        function res = derivShift(self,c,r)
            w = self.weights(:,r);
            psi = self.shifts(:,r);
            
            vc = zeros(self.C,1); vc(c) = 1;
            p = exp(-1i*psi);
            dp = -1i * vc.*p;
            
            res = (w*w') .* (dp*p' + p*dp');
        end
        
        function res = nParams(self)
            res = self.C*self.R + (self.C-1)*self.R;
        end
        
        function keepChannels(self,chans)
            % replace all properties with only the channels in chans
            % NOTE: this will remove data!
            self.C = numel(chans);
            self.logWeights = self.logWeights(chans,:);
            self.shifts = self.shifts(chans,:);
            self.weightsB = self.weightsB(chans,:);
            self.shiftsB = self.shiftsB(chans,:);
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