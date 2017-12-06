function set_seed(i)
% SET_SEED initialization seed for pseudo-random functions
%
% A convinient way to set the seed for the pseudo-random functions
% of MATLAB.

%The function is related on the MATLAB version used
s = RandStream('mcg16807','Seed',i);
ver = version('-release');
if str2double(ver(1:end-1)) < 2013
	RandStream.setDefaultStream(s);
else
	RandStream.setGlobalStream(s);
end

end