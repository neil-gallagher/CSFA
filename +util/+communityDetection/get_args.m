function args = get_args(s)
%GET_ARGS function to support the Community Detection Toolbox
%
%   The function reads the arguments of a matlab function header.
%   The input is a string: 'function out = fun(arg1,arg2)'.
%   The output is a cell array with strings containing {'arg1','arg2'}.

%input validation
if ~ischar(s)
    error('Input must be a string.');
end

if isempty(strfind(s,'(')) || isempty(strfind(s,')')) || ~isscalar(strfind(s,'(')) || ~isscalar(strfind(s,')'))
    error('Invalid input string.');
end

%position of commas (after the first parenthesis bracket) and parenthesis
pos = strfind(s,',');
pos = [strfind(s,'(') pos(gt(pos,strfind(s,'('))) strfind(s,')')];

%number of arguments
nargs = length(pos) - 1;
args  = cell(1,nargs);

for i = 1:nargs
    args{i} = s( pos(i)+1 : pos(i+1)-1 );
end

end