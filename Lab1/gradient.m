function grad = gradient(w, Y, X)
grad = 0;
for i =1:length(Y),
    grad = grad + (Y(i)-sigmoid(X(i,:)*w))*X(i,:);
end