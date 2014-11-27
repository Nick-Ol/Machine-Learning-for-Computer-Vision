function [fts,crds,idxs] =  get_features(input_image,feat,points)
do_dense = (nargin==2);

crds = [];
idxs = [];
%% some pre-processing
if size(input_image,3)>1, input_image = rgb2gray(input_image); end
inim   = single(input_image); inim = inim./max(inim(:));
switch lower(feat),
    case 'sift',
        binSize = 4;
        %% magical number from the documentation of vl_dsift
        magnif  = 3;
        if do_dense
            %% type>>  help vl_dsift  for more justifications
            %%
            Is = vl_imsmooth(inim, sqrt((binSize/magnif)^2 - .25)) ;
            [crds, fts] = vl_dsift(Is, 'size', binSize);
            f(3,:) = binSize/magnif ;
            f(4,:) = 0;
        else
            magnif      = 3;
            f           = [points;[binSize/magnif;0]*ones(1,size(points,2))];
            [crds,fts ] = vl_sift(inim, 'frames', f) ;
        end
        fts = double(fts);
        fts = fts./repmat(max(sqrt(sum(fts.*fts,1)),.01),[128,1]);
    case 'steer',
        filterbank     = construct_filterbank(.8);
        filter_dense   = apply_filterbank(inim,filterbank);
        
        [nfeat,sv,sh]  = size(filter_dense);
        
        if do_dense
            fts            = reshape(filter_dense,[nfeat,sv*sh]);
            [cr_h,cr_v]    = meshgrid([1:sh],[1:sv]);
            crds           = [cr_h(:)';cr_v(:)'];
        else
            npoints = size(points,2);
            fts = zeros(nfeat,npoints);
            for k=1:npoints
                fts(:,k) = filter_dense(:,points(2,k),points(1,k));
            end
        end
end

fts(end+1,:) = 1; %% append the DC term at the end

if do_dense
    [sv,sh] = size(input_image);
    coord_v = crds(2,:);
    coord_h = crds(1,:);
    
    %% matlab indexing
    idxs    = sv*(coord_h-1) + coord_v;
end
