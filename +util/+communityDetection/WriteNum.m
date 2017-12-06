function NumWrite(K,fname)
% function NumWrite(K,fname)
% 
% this function writes number K into file fname
% using integer formatting
%
% INPUT
% K     number to be written
% fname name of the file to which K is written
% 
fid=fopen(fname,'w');
fprintf(fid,'%d\n',K);
fclose(fid);
