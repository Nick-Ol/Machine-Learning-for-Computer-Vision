function hess = hessian(w, Y, X)
hess = 0;
for i =1:length(Y),
	hess = hess + sigmoid(w' * X(i,:)')*(1-sigmoid(w' * X(i,:)'))*(X(i,:)' * X(i,:));
end