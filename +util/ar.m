function [b,B,r,nu,v] = ar(x,p)
% fit a reference posterior AR(p) model to the time series x
% e.g., [coef,prec,innovations,df,variance] = ar(x,p);  
% Uses conditional likelihood - conditional on x_1 
 dat=x-mean(x);
 ndat=length(x);
 y=dat(ndat:-1:p+1);
 n=length(y);
 X=hankel(dat(ndat-1:-1:p),dat(p:-1:1));
 B=X'*X; b=B\(X'*y);
 r=[zeros(p,1);flipud(y-X*b)];
 nu=n-p;
 v=r'*r/nu;