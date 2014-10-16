function [y] = sigmoid(A)
    %y = zeros(size(x));
    %sigmoid for each value of A, which can be a matrix, a vector or a
    %scalar
    y =  1./(1+exp(-A));
    %y =  1./(ones(size(A))+exp(-A));
end
