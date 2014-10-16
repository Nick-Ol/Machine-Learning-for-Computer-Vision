%% Create the training data
nsamples = 200; 
problem  = 'nonlinear';
[train_features_2D,train_labels] = construct_data(nsamples,'train',problem);

%% display your data
pos = find(train_labels==1);
neg = find(train_labels~=1);

hf = figure;
scatter(train_features_2D(1,pos),train_features_2D(2,pos),'r','filled'); hold on,
scatter(train_features_2D(1,neg),train_features_2D(2,neg),'b','filled'); 

%% Apparently the data are not linearly separable. 
%% we therefore use a nonlinear embedding of the features
train_features       = embedding(train_features_2D);
[ndimensions,ndata]  = size(train_features);


%% Some code to visualize the embedding functions
%%
%% evaluate the embedding functions on a regular grid
[grid_x,grid_y] = meshgrid([0:.01:1],[0:.01:1]);
z = embedding([grid_x(:)';grid_y(:)']);
%% show a few of them
figure
subplot(2,2,1);
imshow(reshape(z(3,:),[101,101]),[]);
title('\Phi_3(x_1,x_2)'); xlabel('x_1'); ylabel('x_2');
subplot(2,2,2);
imshow(reshape(z(20,:),[101,101]),[]);
title('\Phi_{20}(x_1,x_2)'); xlabel('x_1'); ylabel('x_2');

subplot(2,2,3);
imshow(reshape(z(40,:),[101,101]),[]);
title('\Phi_{40}(x_1,x_2)'); xlabel('x_1'); ylabel('x_2');

subplot(2,2,4);
imshow(reshape(z(121,:),[101,101]),[]);
title('\Phi_{121}(x_1,x_2)'); xlabel('x_1'); ylabel('x_2');


%% Regularized logistic regresssion training of the resulting classifier 
%% using cross-validation
%%
%% generate candidate regularization coefficients, lambda:
%% geometric progression from 0.0001 to 100, in 20 steps
Nlambdas                = 20;
lambda_range            = logsample(0.0001, 100,Nlambdas);

%% for each of those lambdas
for i=1:Nlambdas
    lambda = lambda_range(1,i);
    
    %% perform K-fold cross validation
    K = 10;
    
    errors  =zeros(1,K);
    fprintf('lambda = %.4f  [%i out of %i]\n',lambda,i,Nlambdas);
    
    for validation_run=1:K
        fprintf('.');
        %% TEMPLATE FOR CROSS-VALIDATION CODE
        
        %split data into training set (trset) and validation set (vlset)
        [trset_features,trset_labels,vlset_features,vlset_labels] =  ...
            split_data(train_features,train_labels,ndata,K,validation_run);
        
        %% train logistic regression @ lambda
        X = trset_features';
        Y = trset_labels';
        
        w = zeros(ndimensions,1); %% initialize w
        
        while 1 %% continue until convergence criterion is met 
            k = k +1;
            w_prev = w;
    
            %% update w (Newton-Raphson)
            J = gradient(w_prev, Y, X, lambda);
            H = hessian(w_prev, Y, X, lambda);
    
            w        = w_prev - pinv(H)*J';
    
            %% convergence criterion
               if sqrt(sum((w-w_prev).^2)/ sqrt(sum(w.^2)))<.1
                  break
               end
        end
        
        predicted_label_test    = (1./(1+exp(-w'*vlset_features)) >.5);
        nerrors(1,validation_run) = length(find(predicted_label_test~=vlset_labels));
   
    end
    fprintf(' \n');
    %The crorss-validation error is the mean of the error
    cv_error(i)=mean(nerrors,2);
end

figure,plot(cv_error);  
print('-depsc','cv_error');

%Pick the lambda that minimizes the cross-validation error

%index of the minimum of cv_error:
index = find(cv_error == min(cv_error(:)));
lambda = lamda_range(index);


%% Retrain using full training set

w = zeros(ndimensions,1); %% initialize w
        
while 1 %% continue until convergence criterion is met 
    k = k +1;
	w_prev = w;
    
	%% update w (Newton-Raphson)
	J = gradient(w_prev, Y, X, lambda);
	H = hessian(w_prev, Y, X, lambda);
    
	w        = w_prev - pinv(H)*J';
    
	%% convergence criterion
	if sqrt(sum((w-w_prev).^2)/ sqrt(sum(w.^2)))<2
        break
	end
end


%% visualize the resulting classifier
dense_score = reshape(w'*z,[101,101]);
figure,
scatter(train_features_2D(1,pos),train_features_2D(2,pos),'r','filled'); hold on,
hold on, 
scatter(train_features_2D(1,neg),train_features_2D(2,neg),'b','filled'); 
hold on,  
contour(grid_x,grid_y,dense_score,[0,0]);
hold on,
axis([0,1,0,1]);
print('-depsc','contours');


%% evaluate performance on test set
[test_features_2D, test_labels ] = construct_data(nsamples,'test', problem);
test_features        = embedding(test_features_2D);

nerrors_test = your_code_goes_here;
