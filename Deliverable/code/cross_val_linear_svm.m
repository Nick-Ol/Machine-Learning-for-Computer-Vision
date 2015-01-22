function [best_cost, cv_error, error] = cross_val_linear_svm(K, cost_range, features, labels)

addpath('util/');
cv_error = zeros(1,size(cost_range,2));
error = zeros(K,size(cost_range,2));
for i=1:size(cost_range,2)
    cost = cost_range(1,i);
    fprintf('C = %.5f  [%i out of %i]\n',cost,i,size(cost_range,2));
    for k=1:K
        [trset_features,trset_labels,vlset_features,vlset_labels] =  ...
                split_data(features,labels,size(features,2),K,k);
         if sum(trset_labels) ~= 0
            [w,model] = linear_svm(trset_features', trset_labels', cost);
            [predict_label, accuracy, dec_values]   ...
                    = svmpredict_libsvm(vlset_labels', vlset_features', model); 
            error(k,i) = length(find(predict_label~=vlset_labels'));
         end
    end
    cv_error(i)=mean(error(:,i));
end

[min_val, min_idx] = min(cv_error);
best_cost = cost_range(min_idx);
