function [coordinate_wl,polarity_wl,theta_wl,err_wl] = ...
    best_weak_learner(Distribution_on_indexes,features,labels)

[d, N] = size(features);
err_wl = inf;

for coordinate = 1:d
    for polarity = -1:2:1 
        for theta_idx = 1:N
            eps = sum(Distribution_on_indexes.*...
                (labels~=decision_stump(polarity, features(coordinate,theta_idx), features(coordinate,:))))...
                /sum(Distribution_on_indexes);
            if eps < err_wl
                coordinate_wl = coordinate;
                polarity_wl = polarity;
                theta_wl = features(coordinate,theta_idx);
                err_wl = eps;
            end
        end
    end
end