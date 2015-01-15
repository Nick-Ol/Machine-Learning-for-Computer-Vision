%% addpaths
this_dir = fileparts(mfilename('fullpath')); addpath(this_dir); addpath(fullfile(this_dir,'steerable'));
run('/Users/Mathurin/Documents/MATLAB/VLFEATROOT/toolbox/vl_setup')

%% take a look into the problem
im_id = 1;
label = 1;
[im,pts] = load_im(im_id,label);
figure,imshow(im);
hold on,scatter(pts(1,:),pts(2,:),'g','filled');

label  = 0;
[im,pts] = load_im(im_id,label);
figure,imshow(im);
hold on,scatter(pts(1,:),pts(2,:),'r','filled');

%%-------------------------------
%% Dataset: train-val-test splits
%--------------------------------

%% images with faces
total_positives = 30;      %% set to 1500 by the end
train_positives = [1:2:total_positives];
test_positives  = [2:2:total_positives];

%% background images (seem few, but we draw multiple negatives samples per image)
total_negatives = 10;       %% set to 200 by the end
train_negatives = [1:2:total_negatives];
test_negatives  = [2:2:total_negatives];


train_set = [train_positives,            train_negatives];
train_lbl = [ones(size(train_positives)),zeros(1,length(train_negatives))];

test_set  = [test_positives,             test_negatives];
test_lbl  = [ones(size(test_positives)), zeros(1,length(test_negatives))];

%%-------------------------------
%% Experiment setup
%--------------------------------

feature_names       = {'SIFT','STEER'};
feature_ind         = 1; %% change to use steerable filter responses
feat                = feature_names{feature_ind};

part_names          = {'Left eye','Right eye','Left mouth','Right mouth','Nose'};

part                = 1; %% change to train classifiers for other parts 
part_name           = part_names{part};

classifier_names    = {'Linear','Logistic','SVM','SVM-RBF','Adaboost'};
classifier          = 1;
classifier_name     = classifier_names{classifier};

%% use this string to save your classifiers 
%experiment_string = sprintf('Features_%s_Part_%s_Classifier_%s',feat,part_name,classifier_name);

%-------------------------------------------------------------------
%% training
%-------------------------------------------------------------------

%% Step 1: gather dataset 
normalize = 1;  % makes sure faces come at a fixed scale
features  = [];
labels    = [];
fprintf('Gathering training set: \n0+ ');

for im_idx = [1:length(train_set)]
    image_id  = train_set(im_idx);
    image_lb  = train_lbl(im_idx);
    
    [input_image,points]    = load_im(image_id,image_lb,normalize,part);
    features_im             = get_features(input_image,feat,points);
    reject = any((isnan(features_im)|isinf(features_im)),1); 
    features_im(:,reject) = [];

    features                = [features,features_im];
    labels                  = [labels,  image_lb*ones(1,size(features_im,2))];
    
    fprintf(2,' %i',mod(im_idx-1,10));
    if mod(im_idx,10)==0, fprintf(2,'\n%i+ ',im_idx); end
end


%% Step 2: train classifier 

switch lower(classifier_name)
    case 'linear'    %% I can do this 
        w_linear = (labels*features')/(features*features');
    case 'logistic'  %% you do the rest 
        w_logistic = log_reg(features', labels')';
        
    case 'adaboost'
        Rounds_boosting = 400;
        Distribution_on_indexes = ones(1,size(features,2))/size(features,2);
        alpha = zeros(1,Rounds_boosting);
        coordinate_wl = zeros(1,Rounds_boosting);
        polarity_wl = zeros(1,Rounds_boosting);
        theta_wl = zeros(1,Rounds_boosting);
        for it = 1:Rounds_boosting,
            [coo,pol,thet,err_wl] = best_weak_learner(Distribution_on_indexes,features,labels);
            coordinate_wl(it) = coo;
            polarity_wl(it) = pol;
            theta_wl(it) = thet;
            % estimate alpha
            alpha(it) = 0.5*log((1-err_wl)/err_wl);    
            % update  distribution on inputs 
            Z = sum(Distribution_on_indexes.*exp(-alpha(it)*labels.*decision_stump(polarity_wl, theta_wl, features(coordinate_wl,:))));
            Distribution_on_indexes = Distribution_on_indexes.*...
                exp(-alpha(it)*labels.*decision_stump(polarity_wl, theta_wl, features(coordinate_wl,:)))...
                /Z;
        end

	case 'svm'
    	addpath('libsvm/');
        parameter_string_lin = sprintf('-s 0 -t 0');
        model_lin = svmtrain_libsvm(labels', features', parameter_string_lin);
        % TODO : cross validation for params ?
	case 'svm-rbf'
        addpath('libsvm/');
        parameter_string_rbf = sprintf('-s 0 -t 2');
        model_rbf = svmtrain_libsvm(labels', features', parameter_string_rbf);
end
if 0
%% fun code: see what the classifier wants to see 
figure,
vl_plotsiftdescriptor(max(w_linear(1:end-1)',0)); 
title('positive components of weight vector');

figure,
vl_plotsiftdescriptor(max(-w_linear(1:end-1)',0));
title('negative  components of weight vector');
end

%-------------------------------------------------------------------
%% testing
%-------------------------------------------------------------------
%% Step 1: gather dataset 
normalize = 1;  % makes sure faces come at a fixed scale
features  = [];
labels    = [];
fprintf('Gathering test set: \n0 +');

for im_idx = [1:length(test_set)]
    image_id  = test_set(im_idx);
    image_lb  = test_lbl(im_idx);
    
    [input_image,points]    = load_im(image_id,image_lb,normalize,part);
    features_im             = get_features(input_image,feat,points);
    
    reject = any((isnan(features_im)|isinf(features_im)),1); features_im(:,reject) = [];   
    features                = [features,features_im];
    
    labels                  = [labels,  image_lb*ones(1,size(points,2))];
    
    fprintf(2,'.%i',mod(im_idx-1,10));
    if mod(im_idx,10)==0, fprintf(2,'\n%i+ ',floor(im_idx/10)); end
end

%% Step 2: Precision recall curve for classifier (your code here);
if 0
thresholds = [-2:.01:2];
for thr_ind  = 1:length(thresholds)
    threshold   = thresholds(thr_ind);
    precision(thr_ind) = your_code;
    recall(thr_ind)    = your_code;
end
plot(precision,recall); 
title_string    = ...
    sprintf('Precision-recall curve for classifier: %s and part: %s',classifier_name,part_name);
title(title_string)
figure;
plot(precision,recall); axis([0,1,0,1]);
title(title_string);
end
%% Step 3: Dense evaluation of classifier
    
for image_lb = [0,1] 
    %% try both a positive and a negative image
    [input_image,points]       = load_im(image_id,image_lb,normalize,part);
    
    %% important: make sure you do NOT give the third, 'points', argument
    [dense_features,crds,idxs] = get_features(input_image,feat);
    
    %% dense_features will be Nfeats X Npoints. For linear classifiers:
    % (adapt accordingly for Adaboost)
    
    switch lower(classifier_name)
        case 'linear',
            score_classifier = w_linear*dense_features;
        case 'logistic'
        case 'svm'
        case 'svm-rbf'
        case 'adaboost'
    end
    [sv,sh]     = size(input_image);
    score       = zeros(sv,sh);
    score(idxs) = score_classifier;
    
    title_string    = sprintf('%s score for part: %s',classifier_name,part_name);
    figure,imagesc(score,[0,.5]); title(title_string);
    figure,imshow(input_image);
end