%% addpaths
this_dir = fileparts(mfilename('fullpath')); addpath(this_dir); addpath(fullfile(this_dir,'steerable'));
run('/Users/Mathurin/Documents/MATLAB/VLFEATROOT/toolbox/vl_setup')
addpath('util/');

%% seeding
rand('seed', 0);

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

% images with faces
total_positives = 1500;      %% set to 1500 by the end
train_positives = [1:2:total_positives];
test_positives  = [2:2:total_positives];

% background images (seem few, but we draw multiple negatives samples per image)
total_negatives = 200;       %% set to 200 by the end
train_negatives = [1:2:total_negatives];
test_negatives  = [2:2:total_negatives];

%%-------------------------------
%% Experiment setup
%--------------------------------

feature_names       = {'SIFT','STEER'};
part_names          = {'Left eye','Right eye','Left mouth','Right mouth','Nose'};
classifier_names    = {'Linear','Logistic','SVM','SVM-RBF','Adaboost'};
classifier          = 4; %% change the classifier here
classifier_name     = classifier_names{classifier};
count_pos = 0;
count_neg = 0;

for feature_ind = 1
    feat = feature_names{feature_ind};
    normalize = 1;  % makes sure faces come at a fixed scale
    
    % We can compute the features for negative labels only once :
    % part=-1 means no part, because negative features
    fprintf('\n Gathering negative training set for feat %s: \n0+ ', feat);
    [features_train_neg, labels_train_neg] = extract_features(train_negatives,feat,0,-1,normalize);
    fprintf('\n Gathering negative test set for feat %s: \n0+ ', feat);
    [features_test_neg, labels_test_neg]  = extract_features(test_negatives,feat,0,-1,normalize);
    
    for part = 1:5
        part_name = part_names{part};

        %% use this string to save your classifiers 
        %experiment_string = sprintf('Features_%s_Part_%s_Classifier_%s',feat,part_name,classifier_name);

        %% Step 1: gather train dataset 
        % positive train
        fprintf('\n Gathering postive train set for %s - %s: \n0+ ',feat,part_name);
        [features_train_pos, labels_train_pos] = extract_features(train_positives,feat,1,part,normalize);

        % Just need to concatenate negative and positive features
        features_train = [features_train_pos, features_train_neg];
        labels_train = [labels_train_pos, labels_train_neg];

        %% Step 2: train classifier 

        switch lower(classifier_name)
            case 'linear'    %% I can do this 
                w_linear = (labels_train*features_train')/(features_train*features_train');
            case 'logistic'  %% you do the rest 
                w_logistic = log_reg(features_train', labels_train')';

            case 'adaboost'
                [alpha,coord,polarity,theta] = adaboost(200, features_train, labels_train);

            case 'svm'
                perm = randperm(size(features_train,2));
                features_perm = features_train(:, perm);
                labels_perm = labels_train(:, perm); % dispatch the labels
                Ncosts  = 40;
                cost_range = logsample(1e-2, 1e3, Ncosts);
                [best_cost_lin, cv_errors, err] = cross_val_linear_svm(7, cost_range, features_perm, labels_perm);
                w_lin_svm = linear_svm(features_train', labels_train', best_cost_lin);

            case 'svm-rbf'
                perm = randperm(size(features_train,2));
                features_perm = features_train(:, perm);
                labels_perm = labels_train(:, perm); % dispatch the labels
                Ngammas = 10;
                Ncosts  = 10;
                gamma_range = logsample(1e-2,1e3,Ngammas);
                cost_range  = logsample(1e-2,1e3,Ncosts);
                [best_cost_rbf, best_gamma] = cross_val_rbf_svm(4, cost_range, gamma_range, features_perm', labels_perm');
                w_rbf_svm = rbf_svm(features_train', labels_train', best_gamma, best_cost_rbf);
        end

        %% fun code: see what the classifier wants to see 
        if 0
        figure,
        vl_plotsiftdescriptor(max(w_linear(1:end-1)',0)); 
        title('positive components of weight vector');

        figure,
        vl_plotsiftdescriptor(max(-w_linear(1:end-1)',0));
        title('negative  components of weight vector');
        end

        %% Gather test dataset 
        fprintf('\n Gathering positive test set for %s - %s: \n0 +',feat,part_name);
        [features_test_pos, labels_test_pos] = extract_features(test_positives,feat,1,part,normalize);

        features_test = [features_test_pos, features_test_neg];
        labels_test = [labels_test_pos, labels_test_neg];

        %% Step 2: Precision recall curve for classifier
        thresholds = [-2:.01:2];
        switch lower(classifier_name)
                case 'linear'
                    [precision, recall] = precision_recall_w(w_linear*features_test, thresholds, labels_test');

                 case 'logistic'
                    [precision, recall] = precision_recall_w(w_logistic*features_test, thresholds, labels_test');

                 case 'svm'
                    [precision, recall] = precision_recall_w(w_lin_svm*features_test, thresholds, labels_test');

                 case 'svm-rbf'
                    [precision, recall] = precision_recall_w(w_rbf_svm*features_test, thresholds, labels_test');

                 case 'adaboost'
                    scores = adaboost_scores(polarity, features_test, coord, theta, alpha);
                    [precision, recall] = precision_recall_w(scores, thresholds, labels_test');

        end


        title_string    = sprintf('%s precision-recall for part: %s',classifier_name,part_name);
        figure;
        plot(precision,recall); axis([0,1,0,1]);
        title(title_string); xlabel('Precision'); ylabel('Recall');

        %% Step 3: Dense evaluation of classifier

        for image_lb = [0,1] 
            %% try both a positive and a negative image
            [input_image,points]       = load_im(200,image_lb,normalize,part);
            % Make sure the image is plotted once only
            if count_pos == 0 && image_lb == 1
                count_pos = 1;
                figure,imshow(input_image);
            end
            if count_neg == 0 && image_lb == 0
                count_neg = 1;
                figure,imshow(input_image);
            end

            %% important: make sure you do NOT give the third, 'points', argument
            [dense_features,crds,idxs] = get_features(input_image,feat);

            %% dense_features will be Nfeats X Npoints. For linear classifiers:
            % (adapt accordingly for Adaboost)

            switch lower(classifier_name)
                case 'linear',
                    score_classifier = w_linear*dense_features;
                case 'logistic'
                    score_classifier = w_logistic*dense_features;
                case 'svm'
                    score_classifier = w_lin_svm*dense_features;
                case 'svm-rbf'
                    score_classifier = w_rbf_svm*dense_features;
                case 'adaboost'
                    score_classifier = adaboost_scores(polarity, dense_features,coord,theta, alpha);
            end
            [sv,sh]     = size(input_image);
            score       = -inf*ones(sv,sh);
            score(idxs) = score_classifier;

            title_string    = sprintf('%s score for part: %s',classifier_name,part_name);
            figure,imagesc(score,[min(score_classifier),max(score_classifier)]); title(title_string);
            
        end
    end
end
