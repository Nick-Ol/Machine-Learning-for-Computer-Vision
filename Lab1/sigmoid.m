function [y] = sigmoid(A)
    %sigmoid for each value of A, which can be a matrix, a vector or a
    %scalar
    y =  1./(1+exp(-A));
end
