function G = arcCosineKernel(U,V,n)
% U is an m-by-p matrix.
% V is an n-by-p matrix.
% n is the family of activation function.
%     n=0 -- step
%     n=1 -- ramp
%     n=2 -- quarter-pipe
% G is an m-by-n Gram matrix of the rows of U and V.

if nargin < 2, V = U; end
if nargin < 3, n = 0; end

Unorm = sqrt(sum(U.^2,2));
Vnorm = sqrt(sum(V.^2,2));

theta = acos(bsxfun(@rdivide,U,Unorm)*bsxfun(@rdivide,V,Vnorm)');

switch n
    case 0
        J = pi - theta;
    case 1
        J = sin(theta) + (pi - theta)*cos(theta);
    case 2
        J = 3*sin(theta)*cos(theta) + (pi - theta)*(1 + 2*cos(theta)^2);    
end

G = 1/pi*bsxfun(@times,Unorm,Vnorm').^n .* J;

end