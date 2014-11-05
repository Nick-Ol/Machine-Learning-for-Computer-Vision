function [weak_learner_output]  = evaluate_stump(features,coordinate_wl,polarity_wl,theta_wl);

feature_slice = features(coordinate_wl,:);
if polarity_wl==1
    crit = feature_slice>=theta_wl;
else
    crit = feature_slice<=theta_wl;
end
weak_learner_output = 2*double(crit) - 1;
