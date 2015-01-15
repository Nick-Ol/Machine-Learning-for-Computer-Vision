function [w] = log_reg(features, labels)
w = zeros(size(features,2),1);
        while 1 %% continue until convergence criterion is met 
            w_prev = w;

            % update w (Newton-Raphson)
            J = gradient(w_prev, labels, features, 0);
            H = hessian(w_prev, features, 0);
            w = w_prev - H\J';
  
            % convergence criterion
            if sqrt(sum((w-w_prev).^2)/ sqrt(sum(w.^2)))<.001
                break
            end
        end