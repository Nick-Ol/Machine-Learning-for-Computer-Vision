function [ features, labels ] = extract_features_scale( indices,feat,image_lb,part,scales )
    features = [];
    labels = [];
    for im_idx = [1:length(indices)]
        image_id  = indices(im_idx);
        for scale = scales
            [input_image,points]    = load_im_scale(image_id,image_lb,scale,part);
        features_im             = get_features(input_image,feat,points);
        reject = any((isnan(features_im)|isinf(features_im)),1); 
        features_im(:,reject) = [];

        features                = [features,features_im];
        labels                  = [labels,  image_lb*ones(1,size(features_im,2))];
        end
        fprintf(2,' %i',mod(im_idx-1,10));
        if mod(im_idx,10)==0, fprintf(2,'\n%i+ ',im_idx); end
    end
end
