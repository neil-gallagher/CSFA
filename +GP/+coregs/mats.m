classdef mats < handle
    % Holds Q coregionalization matrices for C channels, each with rank R_q
    
    properties
        % model parameters
        Q % number of kernels
        C % number of channels
        
        B % cell array of coregionalization matrices
    end
    
    methods
        function self = mats(C,Q,R,B)
            if nargin > 0
                self.Q = Q;
                self.C = C;
                
                if numel(R) == 1, R = repmat(R,[Q,1]); end
                
                if ~exist('B','var')
                    self.B = cell(Q,1);
                    for q = 1:Q
                        self.B{q} = GP.coregs.matComplex(C,R(q));
                    end
                else
                    self.B = reshape(B,[Q,1]);
                end
            end
        end
        
        function res = getParams(self,q)
            if exist('q','var')
                res = self.B{q}.getParams;
            else
                res = cellfun(@(x)x.getParams,self.B,'un',0);
                res = cell2mat(res);
            end
        end
        
        function res = nParams(self)
            res = numel(self.getParams);
        end
        
        function [lb,ub] = getBounds(self)
            [lb,ub] = cellfun(@(x)x.getBounds,self.B,'un',0);
            lb = cell2mat(lb);
            ub = cell2mat(ub);
        end
        
        function qvals = getKernelNums(self)
            qvals = cellfun(@(x,q) q*ones(x.nParams,1),self.B, ...
                num2cell(1:self.Q)','un',0);
            qvals = vertcat(qvals{:});
        end
        
        function setParams(self,params)
            ind = 1;
            for q = 1:self.Q
                nPq = self.B{q}.nParams;
                self.B{q}.setParams(params(ind:ind+nPq-1));
                ind = ind+nPq;
            end
        end
        
        function res = getMat(self,q)
            res = self.B{q}.getMat;
        end
        
        function res = getMatsVec(self)
            res = arrayfun(@(q)self.getMat(q),1:self.Q,'un',0);
            res = cellfun(@(B)B(:),res,'un',0);
            res = vertcat(res{:});
        end
        
        function res = deriv(self)
            res = cellfun(@(x)x.deriv,self.B,'un',0);
            res = vertcat(res{:});
        end
        
        function keepChannels(self,chans)
            % replace all properties with only the channels in chans
            % NOTE: this will remove data!
            self.C = numel(chans);
            for q = 1:self.Q
                self.B{q}.keepChannels(chans);
            end
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
        function res = ind(R,r)
            res = zeros(R,1); res(r) = 1;
        end
    end
end
