classdef kernels < handle
    % Stores Q kernels for multi-output Gaussian processes
    
    properties
        % model parameters
        Q % Number of kernels
        k % Cell array of Q kernels
    end
    
    methods
        function self = kernels(Q,k)
            if nargin > 0
                self.Q = Q;
                if ~exist('k','var')
                    for q = 1:Q
                        self.k{q} = GP.kernels.SG();
                    end
                else
                    self.k = k;
                end
            end
        end
        
        function res = getParams(self,q)
            if nargin<2
                res = cellfun(@(x)x.getParams,self.k,'un',0);
                res = cell2mat(res');
            else
                res = self.k{q}.getParams;  
            end
        end
        
        function [lb,ub] = getBounds(self)
            [lb,ub] = cellfun(@(x)x.getBounds,self.k,'un',0);
            lb = cell2mat(lb');
            ub = cell2mat(ub');
        end
        
        function res = nParams(self)
            res = numel(self.getParams);
        end
        
        function qvals = getKernelNums(self)
            qvals = cellfun(@(x,q) q*ones(x.nParams,1),self.k, ...
                num2cell(1:self.Q),'un',0);
            qvals = vertcat(qvals{:});
        end
        
        function setParams(self,params)
            ind = 1;
            for q = 1:self.Q
                nPq = self.k{q}.nParams;
                self.k{q}.setParams(params(ind:ind+nPq-1));
                ind = ind+nPq;
            end
        end
        
        function res = covFun(self,tau,q)
            if nargin<3 
                res = zeros(numel(tau),self.Q);
                for q = 1:self.Q, res(:,q) = self.k{q}.covFun(tau); end
            else
                res = self.k{q}.covFun(tau);
            end
        end
        
        function res = specDens(self,s,q)
            if nargin<3
                res = zeros(numel(s),self.Q);
                for q = 1:self.Q, res(:,q) = self.k{q}.specDens(s); end
            else
                res = self.k{q}.specDens(s); 
            end
        end
        
        function plotSpecDens(self,minFreq,maxFreq)
            s = linspace(minFreq,maxFreq,1e3);
            plot(s,self.specDens(s),'LineWidth',2);
            xlabel('Frequency','FontSize',14)
        end
        
        function res = derivCovFun(self,tau)
            res = cellfun(@(x)x.derivCovFun(tau),self.k,'un',0);
            res = vertcat(res{:});
        end
        
        function res = derivSpecDens(self,s)
            res = cellfun(@(x)x.derivSpecDens(s),self.k,'un',0);
            res = vertcat(res{:});
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
end
