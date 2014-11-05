function     [weak_learner_on_grid] = evaluate_stump_on_grid(h_range,v_range,coordinate_wl,polarity_wl,theta_wl);

[grh,grv] = meshgrid(h_range,v_range);
if coordinate_wl==1
    feature_slice= grh;
else
    feature_slice = grv;
end

if polarity_wl==1
    crit = feature_slice>=theta_wl;
else
    crit = feature_slice<=theta_wl;
end
weak_learner_on_grid = 2*double(crit) - 1;
