function [hess] = hessian(w, X, lambda)

%we compute the gradient according to the course formula
R = diag(sigmoid(X*w).*(1-sigmoid(X*w)));
hess = -X'*R*X - 2*lambda.*eye(length(X(1,:)));