function hess = hessian(w, Y, X)
hess = 0;
for i =1:length(Y),
	hess = hess - sigmoid(X(i,:)*w)*(1-sigmoid(X(i,:)*w))*(X(i,:)' * X(i,:));
end