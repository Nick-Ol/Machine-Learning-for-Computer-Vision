function [ labels, distortions, iter ] = kmeans( features, K )

[dim, nbPoints] = size(features);
labels = zeros(1, nbPoints)';
prevLabels = ones(1, nbPoints)';
centroids = zeros(dim, K);

%initializing
for i = 1:K 
   centroids(:, i) = features(:, randi(nbPoints));
end
iter = 1;

%looping while labels differ from one iteration to the next
while (prevLabels ~= labels)
    %computing distances
    distances = zeros(nbPoints, K);
    for i = 1:nbPoints
        for j = 1:K
            distances(i, j) = norm(features(:, i) - centroids(:, j));
        end
    end

    %assigning labels
    prevLabels = labels;
    [val, labels] = min(distances, [], 2);

    distortions(iter) = sum(val);
    iter = iter + 1;
end




end

