function [ features, labels ] = extract_features( indices,feat,image_lb,part,normalize )
    features = [];
    labels = [];
    for im_idx = [1:length(indices)]
        image_id  = indices(im_idx);
        if part == -1
            [input_image,points]    = load_im(image_id,image_lb,normalize);
        else
            [input_image,points]    = load_im(image_id,image_lb,normalize,part);
        end
        features_im             = get_features(input_image,feat,points);
        reject = any((isnan(features_im)|isinf(features_im)),1); 
        features_im(:,reject) = [];

        features                = [features,features_im];
        labels                  = [labels,  image_lb*ones(1,size(features_im,2))];

        fprintf(2,' %i',mod(im_idx-1,10));
        if mod(im_idx,10)==0, fprintf(2,'\n%i+ ',im_idx); end
    end


end

