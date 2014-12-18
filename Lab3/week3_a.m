%% Create the training data
nsamples = 500; 
problem  = 'nonlinear';
[features,labels] = construct_data(nsamples,'train',problem,'plusminus');


%% Display your data
pos = find(labels==1);
neg = find(labels~=1);
%splitting postives and negatives
pos_feat = features(:, pos);
neg_feat = features(:, neg);

hf = figure;
scatter(features(1,pos),features(2,pos),'r','filled'); hold on,
scatter(features(1,neg),features(2,neg),'b','filled'); 

%% Run Kmeans on positive and negative data
K = 3; % number of clusters
nbIt = 20; % number of kmeans to run

rng('default') %seeding for reproducibility
pos_init = zeros(nbIt, K); %initial indices for centroids
neg_init = zeros(nbIt, K); %same 
for iter = 1:nbIt
    pos_init(iter, :) = randperm(250, 3);
    neg_init(iter, :) = randperm(250, 3);
end

pos_distortions = zeros(1, nbIt);
neg_distortions = zeros(1, nbIt);
for iter = 1:nbIt
   [pos_labels, pos_centroids, pos_dist] = K_means(pos_feat, K, pos_init(iter, :));
   pos_distortions(iter) = pos_dist(length(pos_dist));
   [neg_labels, neg_centroids, neg_dist] = K_means(neg_feat, K, neg_init(iter, :));
   neg_distortions(iter) = neg_dist(length(neg_dist));
end
%which init gives minimal distortions ?
[val, neg_best] = min(neg_distortions);
[val, pos_best] = min(pos_distortions);

[pos_labels, pos_centroids, pos_dist] = K_means(pos_feat, K, pos_init(pos_best, :));
[neg_labels, neg_centroids, neg_dist] = K_means(neg_feat, K, neg_init(neg_best, :));

%plot distortion as a function of iterations
figure();
hold on
plot(pos_dist)
plot(neg_dist)
xlabel('Number of k-means iterations');
ylabel('Distortion');
legend('Positive features', 'Negative features')
hold off

% plot clusters
colors = ['g', 'r', 'b']
figure()
for i = 1:3
    hold on
    in_cluster_i = find(pos_labels==i);
    scatter(pos_feat(1, in_cluster_i), pos_feat(2, in_cluster_i), colors(i));
    scatter(pos_centroids(1, i), pos_centroids(2, i), 'black', 'fill');
end
figure()
for i = 1:3
    hold on
    in_cluster_i = find(neg_labels==i);
    scatter(neg_feat(1, in_cluster_i), neg_feat(2, in_cluster_i), colors(i));
    scatter(neg_centroids(1, i), neg_centroids(2, i), 'black', 'fill');
end


%% EM
[pos_mu, pos_sigma, pos_clusters, pos_p, pos_LLH] = EM(pos_feat, pos_centroids, 3);
[neg_mu, neg_sigma, neg_clusters, neg_p, neg_LLH] = EM(neg_feat, neg_centroids, 3);
figure()
hold on
plot(pos_LLH)
plot(neg_LLH)
xlabel('Number of Expectation-Maximization iterations');
ylabel('Log-likelihood');
legend('Positive features', 'Negative features')


%plot gaussians
x = -0.1:.01:1.1;
y = -0.1:.01:1.1;

[X Y] = meshgrid(x,y);
figure()
colors = ['g', 'r', 'b']
for i = 1:3
    in_cluster_i = find(pos_clusters==i);
    scatter(pos_feat(1, in_cluster_i), pos_feat(2, in_cluster_i), colors(i))
    mu = pos_mu{i}'; 
    sigma = pos_sigma{i}; 
    Z = mvnpdf([X(:) Y(:)],mu,sigma); 
    Z = reshape(Z,size(X));
    hold on
    contour(X,Y,Z), axis equal  
end
figure();
for i = 1:3
    in_cluster_i = find(neg_clusters==i);
    scatter(neg_feat(1, in_cluster_i), neg_feat(2, in_cluster_i), colors(i))
    mu = neg_mu{i}'; 
    sigma = neg_sigma{i};  
    Z = mvnpdf([X(:) Y(:)],mu,sigma); 
    Z = reshape(Z,size(X));
    hold on
    contour(X,Y,Z), axis equal  
end

%% Boundary decisions
coords = [X(:)';Y(:)'];
posterior = zeros(size(coords, 2), 1);
for i = 1:3
    posterior = posterior + pos_p(i) * mvnpdf(coords', pos_mu{i}', pos_sigma{i}) ...
          - neg_p(i) * mvnpdf(coords', neg_mu{i}', neg_sigma{i});
end
boundary = reshape(posterior, [length(x), length(y)]);
figure()
scatter(features(1,pos),features(2,pos),'r','filled'); hold on,
scatter(features(1,neg),features(2,neg),'b','filled'); hold on,
contour(X, Y, boundary, [0,0])