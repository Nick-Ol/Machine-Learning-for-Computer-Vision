function res = apply_filterbank(input_image,filterbank,chosen_indexes);
%
% res = apply_filterbank(input_image,filterbank);
% applies steerable filterbank to image
% as in the paper of Freeman & Adelson
%
% takes as input a grayscale image and a filterbank computed by construct_filterbank.m
% outputs a cell aray with filter responses
%
% implementation by Iasonas Kokkinos iasonas.kokkinos@ecp.fr
%

filt_germs = filterbank.filt_germs;
lengthtypes = filterbank.ltw;
scales      = filterbank.scales;
scale_filt  = filterbank.scale_filt;
filters     = filterbank.filters;

types_wanted = filterbank.types_wanted;
type_names   = filterbank.type_names;
steering_function = filterbank.steering_function;
nfilts       = length(filterbank.indexes);
n_d          = filterbank.n_d;
log_size     = filterbank.log_size;
[sz_v,sz_h]  = size(input_image);

for scale_index = 1:(filterbank.scales+1)
    for basis_index = 1:length(filt_germs(:,scale_index)),
        if ~isempty(filt_germs{basis_index,scale_index}),
            padding_size = (length(filt_germs{basis_index,scale_index})+1)/2;
            %padding_size = 0;
            tm = filter1(filt_germs{basis_index,scale_index},input_image,1,padding_size);
            filtered_with_germ{basis_index,scale_index} = tm;
        end
    end
end


for scale = 1:scales,
    res_sc = (filter1(filt_germs{1,scale},filtered_with_germ{1,scale},2) -...
        filter1(filt_germs{1,scale+1},filtered_with_germ{1,scale+1},2)); 
    res(scale,:,:) = res_sc(2:end-1,2:end-1);
    if scale==1, res(51,1,1) = 0; end
    % filter_gauss_x, filter_gauss_xx, etc
    combination = filters{scale+1}.combination;
    types_output = [types_wanted];
    for type = types_output,
        dirs =[1:n_d];
        potential_indexes = log_size + (lengthtypes)*(scale-1)*n_d + (dirs-1)*lengthtypes + type;
        if ~strcmp(type_names{type},'qp')
            combo  = combination{type};
            %% length needed for padding
            padding_size = (length(filt_germs{basis_index,scale})+1)/2;
            %padding_size = 0;
            for basis_index = 1:size(combo,1),
                filtered{type,basis_index} = ...
                    filter1(filt_germs{combo(basis_index,1),combo(basis_index,2)},...
                    filtered_with_germ{combo(basis_index,3),combo(basis_index,4)},2,padding_size);
            end
            
            for dir = 1:n_d,
                angle = pi*(dir-1)/n_d; tmp = 0;
                for k = 1:size(combo,1),
                    tmp = tmp + filtered{type,k}*feval(steering_function{type},k,angle);
                end
                idx_filt = log_size + (lengthtypes)*(scale-1)*n_d + (dir-1)*lengthtypes + type;
                res(idx_filt,:,:) = tmp(2:end-1,2:end-1);
            end
        else
            %qp ouutputs
            for dir = 1:n_d,
                index_even =  log_size + (scale-1)*lengthtypes*n_d + (dir-1)*lengthtypes + 2;
                index_odd  =  log_size + (scale-1)*lengthtypes*n_d + (dir-1)*lengthtypes + 3;
                index_qp   =  log_size + (scale-1)*lengthtypes*n_d + (dir-1)*lengthtypes + type;
                tm = sqrt(pow_2(res(index_even,:,:)) + pow_2(res(index_odd,:,:)));
                res(index_qp,:,:) = tm; 
            end
        end
    end
end



function res = pow_2(in);
res = in.*in;

function res= filter1(filt,input,dim,patch_size);
[s_v,s_h] = size(input);
npad =  ceil((length(filt)+1)/2);

if dim==1,
    padsize = [npad,1];
    input   = padarray(input,padsize,'symmetric','both');
    res  = conv2(filt,1,input,'same');
    res    = res(npad+[1:s_v],:);
else
    padsize = [1,npad]; 
    input   = padarray(input,padsize,'symmetric','both');
    res  = conv2(1,filt,input,'same');
    res    = res(:,npad+[1:s_h]);
end
