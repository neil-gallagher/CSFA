classdef SG < handle
    % The Spectral Gaussian (SG) covariance kernel
    
    properties
        % model parameters
        mu % mean of spectral mixture basis
        logVar % variance of spectral mixture basis
        
        % bounds on moddel parameters
        muB % upper/lower bounds on mean term
        logVarB % upper/lower bounds on variance term
        
        nParams % number of parameters in this kernel
    end
    
    methods
        function self = SG(mu,var,muB,varB)
            if nargin > 0
                self.mu = mu;
                self.logVar = log(var);
                self.muB = muB;
                self.logVarB = log(varB);
            else
                self.mu = 2 + gamrnd(10,1);
                self.logVar = log(1+gamrnd(2,2));
                self.muB = [0,500];
                self.logVarB = log([1e-3,500]);
            end
            self.nParams = 2;
        end
        
        function res = getParams(self)
            res = [self.mu; self.logVar];
        end
        
        function setParams(self,params)
            self.mu = params(1);
            self.logVar = params(2);
        end
        
        function res = var(self)
            res = exp(self.logVar);
        end
        
        function [lb,ub] = getBounds(self)
            lb = [self.muB(1); log(exp(self.logVarB(1)) + 5e-2*max(self.mu-20,0))];
            ub = [self.muB(2); self.logVarB(2)];
        end
        
        function res = derivCovFun(self,tau)
            res{1,1} = self.derivCovFunMean(tau);
            res{2,1} = self.derivCovFunLogVar(tau);
        end
        
        function res = derivSpecDens(self,s)
            res{1,1} = self.derivSpecDensMean(s);
            res{2,1} = self.derivSpecDensLogVar(s);
        end
        
        function res = covFun(self,tau)
            res = exp(-2*pi^2*tau.^2*self.var + 2i*pi*tau*self.mu);
        end
        
        function res = specDens(self,s)
            res = 1/sqrt(2*pi*self.var)*exp(-1/2/self.var*(self.mu-s).^2);
        end
        
        function res = derivCovFunLogVar(self,tau)
            res = -2*pi^2*tau.^2*self.var .* self.covFun(tau);
        end
        
        function res = derivCovFunMean(self,tau)
            res = 2i*pi*tau .* self.covFun(tau);
        end
        
        function res = derivSpecDensLogVar(self,s)
            res = 1/2*(1/self.var*(s-self.mu).^2 - 1) .* self.specDens(s);
        end
        
        function res = derivSpecDensMean(self,s)
            res = -1/self.var*(self.mu-s) .* self.specDens(s);
        end
        
                % Make a copy of a handle object.
        function new = copy(self)
            % Instantiate new object of the same class.
            new = feval(class(self));
            
            % Copy all non-hidden properties.
            p = properties(self);
            for i = 1:length(p)
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
