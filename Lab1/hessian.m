function hess = hessian(w, Y, X, lambda)
hess = 0;
for i =1:length(Y),
	hess = hess - sigmoid(X(i,:)*w)*(1-sigmoid(X(i,:)*w))*(X(i,:)' * X(i,:));
end
%and for l2 regularization we add :
hess = hess - 2*lambda.*eye(length(X(1,:)));