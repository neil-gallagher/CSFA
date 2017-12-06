function out = gaborFilter(in,delta,freq,var)
%This function was created by Aditya Sundar 
%The gabor filter uses Gaussian kernel 
% 'in' is the input signal
% 'sigma' is the value of variance
% 'freq' is the centre frequency of the filter 
% contact me at aditsundar@gmail.com for any queries 

%figure(1)
%plot(in)

x = -2:delta:2;
gab = 1/sqrt(2*pi*var)*exp(-1/2*var*x.^2).*exp(1i*freq*x);

out= conv2(in,gab); %Output of filter

%figure(2) 
%plot(abs(out))  %magnitude response of output


end
