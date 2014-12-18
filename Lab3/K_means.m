function [ centroids, distortions ] = K_means( features, K, initInd )

[dim, nbPoints] = size(features);
labels = zeros(1, nbPoints)';
prevLabels = ones(1, nbPoints)';
centroids = zeros(dim, K);
distortions = zeros(1, 1);
%initializing centroids
for i = 1:K 
   centroids(:, i) = features(:, initInd(i));
end
iter = 1;

%looping while labels differ from one iteration to the next
while (not(isequal(prevLabels, labels)))
    %computing distances
    distances = zeros(nbPoints, K);
    for i = 1:nbPoints
        for j = 1:K
            distances(i, j) = norm(features(:, i) - centroids(:, j))^2;
        end
    end

    %assigning labels
    prevLabels = labels;
    [val, labels] = min(distances, [], 2);
    distortions(iter) = sum(val);

    %updating centroids
    for i = 1:K
        indices_in_i = find(labels==i);
        centroids(:, i) = mean(features(:, indices_in_i'), 2); %means over row
    end
    iter = iter + 1;
    
end





end

