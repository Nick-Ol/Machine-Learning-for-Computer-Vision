function[precision, recall] = precision_recall_w(w, threshold, features, labels)

predict_labels = (w*features' >= threshold);

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
precision = TP/(TP+FP);
recall = TP/(TP+FN);
end
