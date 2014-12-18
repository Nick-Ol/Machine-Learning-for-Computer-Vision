function [ mu, sigma, clusters, p ] = EM( features, init_means, k )

[dim, n] = size(features);
mu = cell(1, k);
sigma = cell(1, k);
p = zeros(1, k);
eta = zeros(n, k); %responsibilities
clusters = zeros(n, 1);

for i = 1:k
    mu{i} = init_means(:, i);
    p(i) = 1/k;
    sigma{i} = cov(features');
end
iterations = 0;
LogLikeOld = 10;
LogLikeNew = 0;

while abs(LogLikeNew - LogLikeOld) > 0.1 && iterations < 500
    % E step
    for i = 1:k
        eta(:, i) = mvnpdf(features', mu{i}', sigma{i});
    end
    for l = 1:n
        [val, clusters(l)] = max(eta(l, :)); %attributing points to clusters
        eta(l, :) = eta(l, :)/sum(eta(l, :));
    end
    for i = 1:k
        p(i) = length(find(clusters==i))/n;
    end
    
    % M step
    for i = 1:k
        in_cluster_i = find(clusters==i); % TODO this is computed in the next loop
        %new means
        mu{i} = mean(features(:, in_cluster_i), 2);
        %new covariances
        sigma{i} = cov(features(:, in_cluster_i)');
    end
    LogLikeOld = LogLikeNew;
    LogLikeNew = 0;
    for i = 1:k
         in_cluster_i = find(clusters==i);
         LogLikeNew = LogLikeNew + sum(p(i) * mvnpdf(features(:, in_cluster_i)', mu{i}', sigma{i}));
    end
    iterations = iterations + 1;
    
end
    

end

