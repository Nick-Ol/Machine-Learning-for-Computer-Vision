function grad = gradient(w, Y, X, lambda)
grad = 0;
for i =1:length(Y),
    grad = grad + (Y(i)-sigmoid(X(i,:)*w))*X(i,:);
end
%and for l2 regularization we add :
grad = grad + 2*lambda.*w';