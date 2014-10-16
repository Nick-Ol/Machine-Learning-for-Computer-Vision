%% linear function visualization        
w_linear        = [.1,-.1,.02];
[grid_x,grid_y] = meshgrid([-4:.01:4],[-4:.01:4]);

function_values = w_linear(1)*grid_x + w_linear(2)*grid_y + w_linear(3);

figure,
subplot(1,2,1);
imshow(function_values,[-.1,.1]);
subplot(1,2,2);
mesh(function_values);

figure,
subplot(1,2,1);
imshow(function_values,[]);
subplot(1,2,2);
[d,h] = contour(grid_x,grid_y,function_values,[0,0]);
set(h,'color','r');     % make contour red


%% construct a few sinusoidal basis elements

%% frequencies
freqs = pi*[[0,1/16];... % lower frequencies 
            [1/16,0];...
            [0,1/8]; ...
            [1/8, 0];...
            [0,1/4];...
            [1/4,0];...
            [0,1/2];...
            [1/2,0];...
            [0,1];  ... % higher frequencies
            [1,0]];

%% basis elements
nfreqs  = size(freqs,1);    
for k=1:nfreqs   
    freq_k = freqs(k,:);
    basis_even(:,:,k) = cos(freq_k(1)*grid_x + freq_k(2)*grid_y);
    basis_odd(:,:,k)  = sin(freq_k(1)*grid_x + freq_k(2)*grid_y);
end

%% show elements corresponding to first frequency
figure,
subplot(1,2,1);
imshow(basis_even(:,:,1),[]);
subplot(1,2,2);
imshow(basis_odd(:,:,1),[]);

%% show elements corresponding to final frequency
figure,
subplot(1,2,1);
imshow(basis_even(:,:,end),[]);
subplot(1,2,2);
imshow(basis_odd(:,:,end),[]);


%% construct basis 
basis(:,:,1:nfreqs)             = basis_even;
basis(:,:,nfreqs + [1:nfreqs])  = basis_odd;
size_basis                      = size(grid_x);
basis(:,:,2*nfreqs +  1)        = zeros(size_basis); %% DC component

randn('seed',0); %% ensure same random numbers will be generated at each run
vector = randn(1,2*nfreqs+1);   %% now generate them 

function_values = 0;
for k=1:length(vector)
    function_values = function_values + vector(k)*basis(:,:,k);
end

figure,
subplot(1,2,1);
imshow(function_values,[-.1,.1]);
subplot(1,2,2);
mesh(function_values);

figure,
subplot(1,2,1);
imshow(function_values,[]);
subplot(1,2,2);
[d,h] = contour(grid_x,grid_y,function_values,[0,0]);
set(h,'color','r');     % make contour red







