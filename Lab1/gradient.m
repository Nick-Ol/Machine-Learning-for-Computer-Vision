function [grad] = gradient(w, Y, X, lambda)

%we compute the gradient thanks to the course's formula
grad = (Y'-sigmoid(X*w)')*X - 2*lambda*w';