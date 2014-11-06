function [y] = decision_stump(polarity, theta, x)

if x > theta
    y = polarity;
else
    y = -polarity;
end