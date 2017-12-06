function W=PermMat(N)
% function W=PermMat(N)
% 
% Creates an N-by-N permutation matrix W
%
% INPUT
% N     size of permutation matrix
%
% OUTPUT
% W     the permutation matrix: N-by-N matrix with exactly one 1 in every
%       row and column, all other elements equal to 0
%
% EXAMPLE
% W=PermMat(10);
%
W=zeros(N,N);
q=randperm(N);
for n=1:N; 
	W(q(n),n)=1; 
end
