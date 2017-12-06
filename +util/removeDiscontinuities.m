function [x,y] = removeDiscontinuities(x,y)
dxLimit = 200;
x = x(:).'; y = y(:).';
discontinuities = (abs(diff(y)./diff(x))>dxLimit);
x = [x; nan(1,length(x))];
y = [y; nan(1,length(y))];
x(2*find(~discontinuities)) = [];
y(2*find(~discontinuities)) = [];
x = x(:).'; y = y(:).';
%x(isnan(x)) = 0; y(isnan(y)) = 0;
end