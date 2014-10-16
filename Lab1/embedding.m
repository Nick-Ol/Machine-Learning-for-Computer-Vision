function output = embedding(input)

%% simple embedding using subset of harmonic basis
%% input must be 2 x ndata
%% output will be Nfeatures x ndata

%% sample (w_x,w_y) pairs
%[freqs_x,freqs_y] = meshgrid([0:4*pi:40*pi],[0:4*pi:40*pi]);
[freqs_x,freqs_y] = meshgrid([0:pi:10*pi],[0:pi:10*pi]);
freqs = [freqs_x(:),freqs_y(:)];

%% K x ndata
inprod = freqs*input(1:2,:);  %% phase
%% 2K x ndata
output = [cos(inprod);sin(inprod)];
    