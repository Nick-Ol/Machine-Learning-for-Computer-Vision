
%% addpaths
this_dir    = fileparts(mfilename('fullpath')); addpath(this_dir); addpath(fullfile(this_dir,'steerable'));
input_image  = single(rgb2gray(imread('1.jpg')));

%%---------------------------------------------------------
%% Part 1 
%% dense sift features
%%---------------------------------------------------------
[fts,crds,idxs] =  get_features(input_image,'sift');

%% choose a single image point & visualize sift descriptor around it  

figure(1),imshow(input_image/255);
hold on,scatter(crds(1,point),crds(2,point));
vl_plotsiftdescriptor(fts(1:128,point),[crds(:,point);4;0]);
pause;
clf;
end

%% show elements of sift descriptor, 'dense mode'
[sv,sh] = size(input_image);

figure,
dims_wt = [1:4]; %% sift dimensions being visualized (out of 128)
cnt = 0;
for dim = [dims_wt]
    response            = zeros(sv,sh);
    response(idxs)      = fts(dim,:);
    cnt = cnt  + 1;
    subplot(2,2,cnt);
    imshow(response,[]); title(sprintf('dimension %i',dim));
end

figure,
dims_wt = [125:128]; %% sift dimensions being visualized (out of 128)
cnt = 0;
for dim = [dims_wt]
    response            = zeros(sv,sh);
    response(idxs)      = fts(dim,:);
    cnt = cnt  + 1;
    subplot(2,2,cnt);
    imshow(response,[]);  title(sprintf('dimension %i',dim));
end

%%---------------------------------------------------------
%% Part 2
%% dense steerable filters 
%%---------------------------------------------------------

filter_dense   = (apply_filterbank(input_image,construct_filterbank(.8)));

figure,imshow(input_image,[]);
%% show the value of the 30-th feature over all pixels
figure,imshow(squeeze(filter_dense(30,:,:)),[]);

figure,
imshow(input_image);
for k=1:9,
    subplot(3,3,k);
    imshow(abs(squeeze(filter_dense(k,:,:))),[]);
end

figure,
imshow(input_image);
for k=1:9,
    subplot(3,3,k);
    imshow(abs(squeeze(filter_dense(k+20,:,:))),[]);
end

