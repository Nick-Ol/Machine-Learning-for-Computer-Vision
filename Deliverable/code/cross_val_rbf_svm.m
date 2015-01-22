function [best_cost, best_gamma] = cross_val_rbf_svm(K, cost_range, gamma_range, features, labels)

addpath('util/');
cv_error=zeros(size(gamma_range,2),size(cost_range,2));

for i=1:size(gamma_range,2)
    gamma = gamma_range(1,i);
    for j=1:size(cost_range,2)
        cost = cost_range(1,j);
        error = zeros(1,K);
        fprintf('gamma = %.2f  [%i out of %i],  C = %.2f  [%i out of %i]\n',gamma,i,size(gamma_range,2),cost,j,size(cost_range,2));
        for k=1:K
            [trset_features,trset_labels,vlset_features,vlset_labels] =  ...
                    split_data(features,labels,size(features,2),K,k);
             if sum(trset_labels) ~= 0
                [w,model] = rbf_svm(trset_features', trset_labels', gamma, cost);
                [predict_label, accuracy, dec_values]   ...
                        = svmpredict_libsvm(vlset_labels', vlset_features', model); 
                error(1,k) = length(find(predict_label~=vlset_labels'));
             end
        end
        cv_error(i) = mean(error);
    end
end

[min_val, min_idx] = min(cv_error(:));
%cost on columns :
best_cost_index = ceil(min_idx/size(cv_error,1));
best_cost = cost_range(best_cost_index);
%gamma on rows :
best_gamma_index = mod(min_idx,size(cv_error,1));
best_gamma = gamma_range(best_gamma_index);
