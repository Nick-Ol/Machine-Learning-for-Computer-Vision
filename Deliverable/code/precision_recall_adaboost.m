function [prec, recall] = precision_recall_adaboost(alpha,coord,polarity,theta, ...
                                        features, labels, thresholds)
    rounds = size(polarity, 2);
    n = size(features, 2);
    scores = zeros(1, n);
    for it = 1:rounds
        scores = scores + alpha(it)*polarity(it).*(features(coord(it), :) - theta(it));
    end
    
    nb_thresh = size(thresholds, 2);
    for t = 1:nb_thresh
        pos = scores > thresholds(t);
        predict_pos = find(predict_labels==1);
        predict_neg = find(predict_labels==0);
        label_pos = find(labels==1);
        label_neg = find(labels==0);

        TP = length(intersect(label_pos,predict_pos));
        FN = length(intersect(label_pos, predict_neg));
        FP = length(intersect(label_neg, predict_pos));

        if TP==0
            precision = 0;
            recall = 0;
            if FP==0
                precision = 1;
            end
            if FN==0
                recall = 1;
            end
        else
        prec(t) = TP/(TP+FP);
        recall(t) = TP/(TP+FN);
        
        prec(t) = 
        recall(t) = 
    end
    
end