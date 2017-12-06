function GraphWrite(A,fname)
% function GraphWrite(A,fname)
% 
% this function writes adj matrix A into file fname
% using integer formatting
%
% INPUT
% A     adj matrix to be written
% fname name of the file to which K is written
% 
N=size(A,1);
fid=fopen(fname,'w');
fprintf(fid,'%d\n',N);
for n1=1:N
  for n2=1:N
    fprintf(fid,'%d ',A(n1,n2));
  end
  fprintf(fid,'\n');
end
fclose(fid);