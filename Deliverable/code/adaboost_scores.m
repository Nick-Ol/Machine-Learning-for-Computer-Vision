function[scores] = adaboost_scores(polarity, features, coord, theta, alpha)
    rounds = size(polarity, 2);
    n = size(features, 2);
    scores = zeros(1, n);
    for it = 1:rounds
        scores = scores + alpha(it)*polarity(it).*(features(coord(it), :) - theta(it));
    end
end