%% Create the training data
nsamples = 500; 
problem  = 'nonlinear';
[features,labels,posterior] = construct_data(nsamples,'train',problem,'plusminus');



%%------------------------------------------------------------------
%% visualize training data and posterior
%%------------------------------------------------------------------
figure,
subplot(1,2,1);
imshow(posterior)
title('P(y=1|X): This is the posterior of  the positive class');
subplot(1,2,2);

pos = find(labels==1);
neg = find(labels~=1);
scatter(features(1,pos),features(2,pos),'r','filled'); hold on,
scatter(features(1,neg),features(2,neg),'b','filled'); 

hold on,axis([0,1,0,1]); axis ij; axis square; legend({'positives','negatives'});
title('These are your training data');

%%------------------------------------------------------------------
%% initialize distribution
%%------------------------------------------------------------------

[nfeatures,npoints] = size(features);
Distribution_on_indexes = ones(1,npoints)/npoints;

Rounds_boosting = 400;
f = zeros(Rounds_boosting,npoints);
f_on_grid = 0;

for it = 1:Rounds_boosting,
    
    %%--------------------------------------------------------
    %% Find best weak learner at current round of boosting
    %%--------------------------------------------------------
    [coordinate_wl,polarity_wl,theta_wl,err_wl] = best_weak_learner(Distribution_on_indexes,features,labels);
    
    %%--------------------------------------------------------
    %% estimate alpha
    %%--------------------------------------------------------
    
    
    
    %%--------------------------------------------------------
    %% update  distribution on inputs 
    %%--------------------------------------------------------
    
    
    %%--------------------------------------------------------
    %% compute loss of adaboost at current round
    %%--------------------------------------------------------
    
    
    
    %% leave as is - it will produce the classifier images for you
    [weak_learner_on_grid] = evaluate_stump_on_grid([0:.02:1],[0:.02:1],coordinate_wl,polarity_wl,theta_wl);

    %%--------------------------------------------------------
    %% add current weak learner's response to overall response
    %%--------------------------------------------------------
    f_on_grid   = f_on_grid + alpha(it).*weak_learner_on_grid;
    switch it
        case 10,
            f_10 = f_on_grid;
        case 50,
            f_50 = f_on_grid;
        case 100
            f_100 = f_on_grid;
    end
    
end

figure,
plot(err)
print('-depsc','loss_adaboost')

figure
subplot(1,2,1);
imshow(posterior);
subplot(1,2,2);
imshow(1./(1+exp(-2*f_on_grid)))

print('-depsc','classifiers')

figure
subplot(1,4,1);
imshow(f_10,[-1,1]); title('strong learner - round 10')
subplot(1,4,2);
imshow(f_50,[-1,1]); title('strong learner - round 50')
subplot(1,4,3);
imshow(f_100,[-1,1]); title('strong learner - round 100')
subplot(1,4,4);
imshow(f_on_grid,[-1,1]); title('strong learner - round 400')
print('-depsc','per_round')
