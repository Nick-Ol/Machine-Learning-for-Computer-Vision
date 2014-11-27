function filterbank = construct_filterbank(factor,extras);
%
% filterbank = make_separable_filterbank(factor,extras);
% constructs steerable filterbank
% as in the paper of Freeman & Adelson
%
% `factor' determines the filter scales
%
% implementation by Iasonas Kokkinos iasonas.kokkinos@ecp.fr
%
%

% default settings
% Filter types:
% 1 derivative,
% 2 second derivative
% 3 hilbert transform of second derivative
% 4 amplitude estimate (sqrt(2.^2 + 3.^2))

nscales = 3; n_d =4; types_wanted = [1:4];

if nargin>1, structure = extras; expand_varargin; end
if nargin==0, factor = 1/4; end
ltw = length(types_wanted);

scales_dog = 2.^[0:nscales]*factor;
scales =  scales_dog(2:end);

second_order = 1;
use_hilbert = 1;   % can be used to derive gabor-like phase-invariant features

steering_function = {'deriv_g_x_orient','deriv_g_xx_orient','deriv_h_xx_orient'};

for i=1:(length(scales_dog)),
    filt_germs{1,i} = kernel_gauss(scales_dog(i));
end

%% computational germs used for the computation of x/y separable functions
%% that are used for steering
for i=1:length(scales),
    filt_germs{2,i} =  kernel_gauss(scales(i));
    filt_germs{3,i} =  kernel_gauss_x(scales(i));
    filt_germs{4,i} =  kernel_gauss_xx(scales(i));
    filt_germs{5,i} =  kernel_hilb_1(scales(i));
    filt_germs{6,i} =  kernel_hilb_2(scales(i));
end

set_count = 1;  %% multi-scale Difference of gaussian filters, first stack of filters
for i=1:length(scales)
    %% store the filter indexes needed to construct a dog filter at scale i
    %% specifically: each bracket contains:
    % [Horizontal_type,Horizontal_scale,Vertical_type,Vertical_scale,sign(optional)]
    filters{set_count}.combination{i}(1,:) = [1,i,  1,i,  1];
    filters{set_count}.combination{i}(2,:) = [1,i+1,1,i+1,-1];
    size_filter = length(filt_germs{1,i+1}); center_f = div(size_filter,2) + 1;
    zero_filter = zeros(size_filter);
    filter_full{set_count}{i} = put_patch(zero_filter,center_f,center_f,kron(filt_germs{1,i},filt_germs{1,i}'))...
        -put_patch(zero_filter,center_f,center_f,kron(filt_germs{1,i+1},filt_germs{1,i+1}'));
    filter_cell{i} =      filter_full{set_count}{i};
    scale_filt(i)  = ceil(5*sqrt(scales(i)/2)); %zsize(filter_cell{i},1)/2;
    indexes(i) = 0;
end
log_size = length(filter_full{1});

for i=1:length(scales),
    set_count = set_count + 1;
    if ismember(1,types_wanted)  % odd difference
        filters{set_count}.combination{1}(1,:) = [3,i,2,i];
        filters{set_count}.combination{1}(2,:) = [2,i,3,i];
    end
    if ismember(2,types_wanted), % even difference
        %e.g. the first steering function is the
        % kron product of gaus & gauss_xx
        % and the second is gauss_x & gaus_y
        filters{set_count}.combination{2}(1,:) = [4,i,2,i];
        filters{set_count}.combination{2}(2,:) = [3,i,3,i];
        filters{set_count}.combination{2}(3,:) = [2,i,4,i];
    end
    if ismember(3,types_wanted)  % hilbert transform:
        %e.g. the first steering function is the
        % kron product of Hilbert_1 and gauss_x (see p. 904, Freeman)
        
        filters{set_count}.combination{3}(1,:) = [5,i,2,i];
        filters{set_count}.combination{3}(2,:) = [6,i,3,i];
        filters{set_count}.combination{3}(3,:) = [3,i,6,i];
        filters{set_count}.combination{3}(4,:) = [2,i,5,i];
    end
    
    for k=1:n_d,
        for type = setdiff(types_wanted,4)
            filter_full{set_count}{ltw*(k-1) + type} =  ...
                make_full_filter(filt_germs,filters{set_count}.combination{type},...
                (k-1)/n_d*pi,steering_function{type});
            filt_ind  = log_size + (ltw)*(i-1)*n_d + (k-1)*ltw + type;
            
            filter_cell{filt_ind} = filter_full{set_count}{ltw*(k-1) + type};
            indexes(filt_ind) = type;
            scale_filt(filt_ind)  = ceil(5*sqrt(scales(i)/2));
            if ismember(4,types_wanted)
                indexes(log_size + (ltw)*(i-1)*n_d + (k-1)*ltw + 4) = 4;
                scale_filt(log_size + (ltw)*(i-1)*n_d + (k-1)*ltw + 4) = ceil(5*sqrt(scales(i)/2));
            end
        end
    end
end
ltw = length(types_wanted);
type_names  ={'g_deriv_x','g_deriv_xx','hilbert_xx','qp'};
scales = length(filter_full) - 1;
hilbert = use_hilbert;

filterbank = struct('log_size',log_size,'n_d',n_d,'hilbert',hilbert,'second_order',second_order,'ltw',ltw);
filterbank.steering_function = steering_function;
filterbank.indexes           = indexes;
filterbank.types_wanted      = types_wanted;
filterbank.type_names        = type_names;
filterbank.scales            = scales;
filterbank.filter_cell       = filter_cell;
filterbank.scale_filt        = scale_filt;
filterbank.filt_germs        = filt_germs;
filterbank.filters           = filters;

%'filters',filters,'filter_full',filter_full,'filt_germs',filt_germs
%compress_structure;
%filterbank =  structure;


function res = kernel_gauss(scale,cnt);
if ~exist('cnt'),
    cnt = 2.4;
end
width = ceil(cnt*sqrt(2)*scale);
limits  =  [-width:width];
values  =  exp( - (limits/scale).^2)*(1/scale);
res = values;

function res = kernel_gauss_x(scale,cnt);
if ~exist('cnt'),
    cnt = 2.4;
end
width = ceil(cnt*sqrt(2)*scale);
limits  =  [-width:width];
limits_scaled = limits/scale;
values_x = (limits_scaled).*exp( - (limits_scaled).^2);
res = values_x;

function res = kernel_gauss_xx(scale,cnt);
if ~exist('cnt'),
    cnt = 2.4;
end
width = ceil(cnt*sqrt(2)*scale);
limits  =  [-width:width];
values_xx = (2*((limits/scale).^2)-1).*exp( - (limits/scale).^2);
res = values_xx;

function res = kernel_hilb_1(scale,cnt);
if ~exist('cnt'),
    cnt = 2.4;
end
width = ceil(cnt*sqrt(2)*scale);
limits  =  [-width:width];

limits_scaled   = limits/scale;
values_xx = (-2.254*limits_scaled + limits_scaled.^3).*exp( - (limits_scaled).^2);
res = values_xx;

function res = kernel_hilb_2(scale,cnt);
if ~exist('cnt'),
    cnt = 2.4;
end
width = ceil(cnt*sqrt(2)*scale);
limits  =  [-width:width];

limits_scaled   = limits/scale;
values_xx = (-.7515 + limits_scaled.^2).*exp( - (limits_scaled).^2);
res  = values_xx;


function image = put_patch(image,f_x,f_y,patch);
[l_x,l_y] = size(patch);
if 2*div(l_x,2)==l_x
    patch =  imresize(patch,[l_x+1,l_x+1]);
    l_x = l_x+1; l_y = l_y+1;
end
[patch_x,patch_y] = size(patch);
patch_x_used = (patch_x-1)/2;
patch_y_used = (patch_y-1)/2;

[size_x,size_y] = size(image);

less_x = abs(min(f_x - patch_x_used-1,0));
less_y = abs(min(f_y - patch_y_used-1,0));

more_x = max((f_x + patch_x_used - size_x),0);
more_y = max((f_y + patch_y_used - size_y),0);

expand = max([less_x,more_x,less_y,more_y]);
if expand>0,
    image = my_patch(image,expand,-1);
    f_x = f_x + expand; f_y = f_y + expand;
end
l_x = l_x-1; l_y  = l_y-1;
image(f_x+ (-floor(l_x/2):floor(l_x/2)),f_y +  (-floor(l_y/2):floor(l_y/2)))...
    = patch;
if expand>0
    image =  peel(image,expand);
end

function res = make_full_filter(separable_filters,wanted_filter_pairs,angle,steering_function)
res = 0;
wanted_filter_pairs = squeeze(wanted_filter_pairs);
for k=1:size(wanted_filter_pairs,1),
    kernel =  kron(separable_filters{wanted_filter_pairs(k,1),wanted_filter_pairs(k,2)},...
        separable_filters{wanted_filter_pairs(k,3),wanted_filter_pairs(k,4)}');
    res = res + kernel*feval(steering_function,k,angle);
end


function res = div(a,b)
res = floor(a/b);