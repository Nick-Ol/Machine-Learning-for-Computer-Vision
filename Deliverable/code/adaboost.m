function[alpha,coordinate_wl,polarity_wl,theta_wl] = adaboost(Rounds, features, labels)
    labelsNeg = 2*labels - 1; 
    Distribution_on_indexes = ones(1, size(features, 2))/size(features, 2);
    alpha = zeros(1, Rounds);
    coordinate_wl = zeros(1, Rounds);
    polarity_wl = zeros(1, Rounds);
    theta_wl = zeros(1, Rounds);
    for it = 1:Rounds
        [coo,pol,thet,err_wl] = best_weak_learner(Distribution_on_indexes,features,labelsNeg);
        coordinate_wl(it) = coo;
        polarity_wl(it) = pol;
        theta_wl(it) = thet;
        % estimate alpha
        if err_wl == 0
            break
        end
        alpha(it) = 0.5*log((1-err_wl)/err_wl);    
        % update  distribution on inputs 
        Z = sum(Distribution_on_indexes.*exp(-alpha(it)*labelsNeg.*decision_stump(pol, thet, features(coo,:))));
        Distribution_on_indexes = Distribution_on_indexes.*...
            exp(-alpha(it)*labelsNeg.*decision_stump(pol, thet, features(coo,:)))...
            /Z;
    end
end