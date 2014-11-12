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
alpha = zeros(1,Rounds_boosting);
err = zeros(1, Rounds_boosting);
nb_err = zeros(1, Rounds_boosting);

%test set :
[test_features,test_labels] = construct_data(nsamples,'test',problem,'plusminus');
f_test = zeros(Rounds_boosting,npoints);

for it = 1:Rounds_boosting,
    
    %%--------------------------------------------------------
    %% Find best weak learner at current round of boosting
    %%--------------------------------------------------------
    [coordinate_wl,polarity_wl,theta_wl,err_wl] = best_weak_learner(Distribution_on_indexes,features,labels);
    
    %%--------------------------------------------------------
    %% estimate alpha
    %%--------------------------------------------------------
    
    alpha(it) = 0.5*log((1-err_wl)/err_wl);    
    
    %%--------------------------------------------------------
    %% update  distribution on inputs 
    %%--------------------------------------------------------
    
    Z = sum(Distribution_on_indexes.*exp(-alpha(it)*labels.*decision_stump(polarity_wl, theta_wl, features(coordinate_wl,:))));
    Distribution_on_indexes = Distribution_on_indexes.*...
        exp(-alpha(it)*labels.*decision_stump(polarity_wl, theta_wl, features(coordinate_wl,:)))...
        /Z;
    
    %%--------------------------------------------------------
    %% compute loss of adaboost at current round
    %%--------------------------------------------------------
    
    f(it,:) = alpha(it)*decision_stump(polarity_wl, theta_wl, features(coordinate_wl,:));
    err(it) = sum(exp(-labels.*sum(f,1)));
    nb_err(it) = sum(labels~=sign(sum(f,1)));
    
    %Value on test set :
    f_test(it,:) = alpha(it)*decision_stump(polarity_wl, theta_wl, test_features(coordinate_wl,:));
    
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
plot(err), hold on,
plot(nb_err)
xlabel('Iterations' )
legend('Exponential loss', 'Number of errors')
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

% Performance on test set
nerrors_adaboost = sum(test_labels~=sign(sum(f_test,1)))
