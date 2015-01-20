function[precision, recall] = precision_recall_w(scores, thresholds, labels)

precision = zeros(1,length(thresholds));
recall= zeros(1,length(thresholds));


for thr_ind  = 1:length(thresholds)
    threshold   = thresholds(thr_ind);
    predict_labels = (scores >= threshold);

    predict_pos = find(predict_labels==1);
    predict_neg = find(predict_labels==0);
    label_pos = find(labels==1);
    label_neg = find(labels==0);

    TP = length(intersect(label_pos,predict_pos));
    FN = length(intersect(label_pos, predict_neg));
    FP = length(intersect(label_neg, predict_pos));

    if TP==0
        precision(thr_ind) = 0;
        recall(thr_ind) = 0;
        if FP==0
            precision(thr_ind) = 1;
        end
        if FN==0
            recall(thr_ind) = 1;
        end
    else
    precision(thr_ind) = TP/(TP+FP);
    recall(thr_ind) = TP/(TP+FN);
    end
end
end
