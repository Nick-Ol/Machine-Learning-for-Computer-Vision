function grad = gradient(w, Y, X, lambda)

grad = (Y'-sigmoid(X*w)')*X - 2*lambda*w';