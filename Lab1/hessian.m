function hess = hessian(w, X, lambda)

R = diag(sigmoid(X*w).*(1-sigmoid(X*w)));
hess = -X'*R*X - 2*lambda.*eye(length(X(1,:)));