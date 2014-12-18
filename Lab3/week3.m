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


K = 3; % number of clusters
nbIt = 10; % number of kmeans run

rng('default')
pos_init = zeros(nbIt, K);
neg_init = zeros(nbIt, K);
for iter = 1:nbIt
    pos_init(iter, :) = randperm(250, 3);
    neg_init(iter, :) = randperm(250, 3);
end

pos_distortions = zeros(1, nbIt);
neg_distortions = zeros(1, nbIt);
for iter = 1:nbIt
   [pos_centroids, pos_distortions(iter)] = K_means(pos_feat, K, pos_init(iter, :));
   [neg_centroids, neg_distortions(iter)] = K_means(neg_feat, K, neg_init(iter, :));
end
min(neg_distortions)
min(pos_distortions)