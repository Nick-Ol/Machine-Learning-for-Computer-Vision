function [y] = decision_stump(polarity, theta, x)

y = polarity*(2*(x>theta)-1);

end