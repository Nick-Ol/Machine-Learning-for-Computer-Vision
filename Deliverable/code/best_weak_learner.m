function [coordinate_wl,polarity_wl,theta_wl,err_wl] = best_weak_learner(Distribution_on_indexes,features,labels)
[nfeatures,npoints] = size(features);

for ft = 1:nfeatures
    [srt,idx]           = sort(features(ft,:));
    [l,ulast,i]         = unique(srt,'last');
    [l,ufirst,i]        = unique(srt,'first');
    labels_sorted       = labels(idx);
    weights_sorted      = Distribution_on_indexes(idx);
    
    
    % say that the k-th sorted value is V_k. Consider that we put the
    % threshold at V_k.
    % if we say all points below V_k are negative, and all above/equal
    % V_k are positive, we lose the positives below V_k and the
    % negatives above V_k.
    % We can compute this recursively in k, by using cumsum
    
    weights_of_positives = (labels_sorted==1).* weights_sorted;
    weights_of_negatives = (labels_sorted~=1).*weights_sorted;
    
    sum_positive_below_thr = [0,cumsum(weights_of_positives)];
    sum_negative_below_thr = [0,cumsum(weights_of_negatives)];
    
    sum_positive_above_thr = sum_positive_below_thr(end) - sum_positive_below_thr(1:end-1);
    sum_negative_above_thr = sum_negative_below_thr(end) - sum_negative_below_thr(1:end-1);
    
    cost_at_threshold_negpos  = sum_positive_below_thr(2:end) + sum_negative_above_thr;
    cost_at_threshold_posneg  = sum_positive_above_thr + sum_negative_below_thr(2:end);
    
    
    [mnp,inp] = min(cost_at_threshold_negpos(ulast));
    [mpn,ipn] = min(cost_at_threshold_posneg(ufirst));
    if mnp<mpn,
        err(ft)         = mnp;
        threshold(ft)   = srt(ulast(inp));
        polarity(ft)    = 1;
    else
        err(ft)         = mpn;
        threshold(ft)   = srt(ufirst(ipn));
        polarity(ft)    = -1;
    end
end
[err_wl,coordinate_wl] = min(err);
polarity_wl = polarity(coordinate_wl);
theta_wl    = threshold(coordinate_wl);

