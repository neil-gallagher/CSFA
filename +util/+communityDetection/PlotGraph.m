function h = PlotGraph2(A,V0)
% function h = PlotGraph2(A,V0)
%
% Plots the graph with adjacency matrix A and partition V0
%
% INPUT
% A     N-by-N adjacency matrix of graph
% V0    N-by-1 vector of integers, shows to which cluster each node belongs
%
% OUTPUT
% None
%
% EXAMPLE
% [A,V0]=GGGN(32,4,16,0,0);
% VV=GCAFG(A,[0.2:0.5:1.5]);
% Kbst=CNModul(VV,A);
% V=VV(:,Kbst);
% PlotGraph1(A,V);
%
mu=10;
sg=5;
clr='bgrmcbgrmcbgrmcbgrmcbgrmc';
clr=[clr clr clr clr clr clr];
clr(3);

N = size(A,1); % number of nodes
N0=[1:N]';
K = max(V0);   % number of clusters
angl = 2*pi/K; % rotation angle

W=PermMat(N); N0=W*N0; V0=V0(N0); A=W*A*W';

for k=1:K
  X(k) = cos(angl*(k-1));
  Y(k) = sin(angl*(k-1));
end

for n=1:N
    x(n,1)=mu*X(V0(n))+sg*rand(1,1);
    y(n,1)=mu*Y(V0(n))+sg*rand(1,1);
end

h = figure('MenuBar','none'); clf;
for n1=1:N
    for n2=1:N
        if A(n1,n2)>0
            line([x(n1) x(n2)],[y(n1) y(n2)],'Color','k','LineStyle','-','LineWidth',1.5); hold on
        end
    end
end
for n=1:N
  plot(x(n),y(n),['ko'],'LineWidth',5,'MarkerSize',12);  hold on
  plot(x(n),y(n),[clr(V0(n)) '*'],'LineWidth',5,'MarkerSize',12);  hold on  
end
h = figure(h); axis off; hold off
