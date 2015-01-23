%% addpaths
this_dir = fileparts(mfilename('fullpath')); addpath(this_dir); addpath(fullfile(this_dir,'steerable'));
%run('/Users/Mathurin/Documents/MATLAB/VLFEATROOT/toolbox/vl_setup')
addpath('util/');

%% seeding
rand('seed', 0);

%% take a look into the problem
im_id = 1;
label = 1;
[im,pts] = load_im_scale(im_id,label,0.5);
figure,imshow(im);
hold on,scatter(pts(1,:),pts(2,:),'g','filled');

label  = 0;
[im,pts] = load_im_scale(im_id,label,0.5);
figure,imshow(im);
hold on,scatter(pts(1,:),pts(2,:),'r','filled');

%%-------------------------------
%% Dataset: train-val-test splits
%--------------------------------

% images with faces
total_positives = 20;      %% set to 1500 by the end
train_positives = [1:2:total_positives];
test_positives  = [2:2:total_positives];

% background images (seem few, but we draw multiple negatives samples per image)
total_negatives = 20;       %% set to 200 by the end
train_negatives = [1:2:total_negatives];
test_negatives  = [2:2:total_negatives];

%%-------------------------------
%% Experiment setup
%--------------------------------

scales = [2^(-1), 2^(-0.5), 1, 2^0.5, 2];
feature_names       = {'SIFT','STEER'};
part_names          = {'Left eye','Right eye','Left mouth','Right mouth','Nose'};
classifier_names    = {'Linear','Logistic','SVM','SVM-RBF'};
classifier          = 1; %% change the classifier here
classifier_name     = classifier_names{classifier};
count_pos = 0;
count_neg = 0;

feature_ind = 1; % chose the feature
feat = feature_names{feature_ind};
normalize = 1;

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
    [features_train_pos, labels_train_pos] = extract_features_scale(train_positives,feat,1,part,scales);

    % Just need to concatenate negative and positive features
    features_train = [features_train_pos, features_train_neg];
    labels_train = [labels_train_pos, labels_train_neg];

    %% Step 2: train classifier 

    switch lower(classifier_name)
        case 'linear'    %% I can do this 
            w_linear = (labels_train*features_train')/(features_train*features_train');
        case 'logistic'  %% you do the rest 
            w_logistic = log_reg(features_train', labels_train')';

        case 'svm'
            perm = randperm(size(features_train,2));
            features_perm = features_train(:, perm);
            labels_perm = labels_train(:, perm); % dispatch the labels
            Ncosts  = 10;
            cost_range = logsample(1e-3,5,Ncosts);
            [best_cost_lin, cv_errors, err] = cross_val_linear_svm(10, cost_range, features_perm, labels_perm);
            w_lin_svm = linear_svm(features_train', labels_train', best_cost_lin);

        case 'svm-rbf'
            perm = randperm(size(features_train,2));
            features_perm = features_train(:, perm);
            labels_perm = labels_train(:, perm); % dispatch the labels
            Ngammas = 10;
            Ncosts  = 10;
            gamma_range = logsample(1e-3,5,Ngammas);
            cost_range  = logsample(1e-3,5,Ncosts);
            [best_cost_rbf, best_gamma] = cross_val_rbf_svm(10, cost_range, gamma_range, features_perm, labels_perm);
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
    [features_test_pos, labels_test_pos] = extract_features_scale(test_positives,feat,1,part,scales);

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

    end


    title_string    = sprintf('%s precision-recall for part %s and %s features',classifier_name,part_name,feat);
    figure;
    plot(precision,recall); axis([0,1,0,1]);
    title(title_string); xlabel('Precision'); ylabel('Recall');

    %% Step 3: Dense evaluation of classifier

    for image_lb = [0,1]
        if image_lb == 0
            %% negative image
        [input_image,points]       = load_im(200,image_lb,normalize,part);
        % Make sure the image is plotted once only
        if count_neg == 0
            count_neg = 1;
            figure,imshow(input_image);
        end

        %% important: make sure you do NOT give the third, 'points', argument
        [dense_features,crds,idxs] = get_features(input_image,feat);

        %% dense_features will be Nfeats X Npoints. For linear classifiers:

        switch lower(classifier_name)
            case 'linear',
                score_classifier = w_linear*dense_features;
            case 'logistic'
                score_classifier = w_logistic*dense_features;
            case 'svm'
                score_classifier = w_lin_svm*dense_features;
            case 'svm-rbf'
                score_classifier = w_rbf_svm*dense_features;
        end
        [sv,sh]     = size(input_image);
        score       = -inf*ones(sv,sh);
        score(idxs) = score_classifier;

        title_string    = sprintf('%s score for part %s and %s features',classifier_name,part_name, feat);
        figure,imagesc(score,[min(score_classifier),max(score_classifier)]); title(title_string);
        else
            %% positive image : different scales
            for scale = scales
                [input_image,points]       = load_im_scale(200,image_lb,scale,part);
                % Make sure the image is plotted once only
                if count_pos == 0 && image_lb == 1
                    count_pos = 1;
                    figure,imshow(input_image);
                end

                %% important: make sure you do NOT give the third, 'points', argument
                [dense_features,crds,idxs] = get_features(input_image,feat);

                %% dense_features will be Nfeats X Npoints. For linear classifiers:

                switch lower(classifier_name)
                    case 'linear',
                        score_classifier = w_linear*dense_features;
                    case 'logistic'
                        score_classifier = w_logistic*dense_features;
                    case 'svm'
                        score_classifier = w_lin_svm*dense_features;
                    case 'svm-rbf'
                        score_classifier = w_rbf_svm*dense_features;
                end
                [sv,sh]     = size(input_image);
                score       = -inf*ones(sv,sh);
                score(idxs) = score_classifier;

                title_string    = sprintf('%s score for part %s and %s features and scale %s',classifier_name,part_name, feat, scale);
                figure,imagesc(score,[min(score_classifier),max(score_classifier)]); title(title_string);
            end
        end
    end
    %% Compute unary terms
    switch lower(classifier_name)
                    case 'linear',
                        weights_unary(part,:) = w_linear;
                    case 'logistic'
                        weights_unary(part,:) = w_logistic;
                    case 'svm'
                        weights_unary(part,:) = w_lin_svm;
                    case 'svm-rbf'
                        weights_unary(part,:) = w_rbf_svm;
    end 
end

%% Now, message passing :

%% Maximum Likelihood parameter estimation for pairwise terms
for im_id=1:1000,
    for scale=scales
        [input_image,points] = load_im_scale(im_id,1,scale);
        center = points(:,5);
        offsets(:,:,im_id) = points(:,1:4) - center*ones(1,4);
    end
end

for pt = [1:4],
    mn{pt} = mean(squeeze(offsets(:,pt,:)),2);
    sg{pt} = sqrt(diag(cov(squeeze(offsets(:,pt,:))')));
end

%% take a look at the data
strs = {'left eye','right eye','left mouth','right mouth','nose'};

clrs = {'r','g','b','k','y'};
figure,
for pt = [1:4],
    scatter(squeeze(offsets(1,pt,:)),squeeze(offsets(2,pt,:)),clrs{pt},'filled'); hold on,
    text(mn{pt}(1),mn{pt}(2),strs{pt},'fontsize',30)
end
axis ij; axis equal;

im_id         = 231;
scale = 2^(-0.5); % can be customized
[input_image] = load_im_scale(im_id,1,scale);
[feats,~,idxs]= get_features(input_image,'SIFT');
responses     = weights_unary*feats;
[sv,sh]       = size(input_image);
for pt_ind = [1:5],
    score       = -10*ones(sv,sh);
    score(idxs) = responses(pt_ind,:);
    score_part{pt_ind} = score;
end

figure
subplot(2,3,1); imshow(input_image);
for pt_ind = [1:5],
    subplot(2,3,1+pt_ind);
    imshow(score_part{pt_ind},[-2,2]);
    title([strs{pt_ind},' with SVM-Linear - SIFT']);
end

%% dt- potential:  def(1) h^2 + def(2) h + def(3) * v^2 + def(4) *v
%% gaussian potential:   (h - mh)^2/(2*sch^2) + (v-mv)^2/(2*scv^2)
for pt = [1:4]
    sch = sg{pt}(1);
    scv = sg{pt}(2);
    mh  = -mn{pt}(1);
    mv  = -mn{pt}(2);
    
    def(1) = 1/(2*sch^2);
    def(2) = -2*mh/(2*sch^2);
    def(3) = 1/(2*scv^2);
    def(4) = -2*mv/(2*scv^2);
    
    [mess{pt},ix{pt},iy{pt}] = dt(squeeze(score_part{pt}),def(1),def(2),def(3),def(4));
    offset =  mh^2/(2*sch^2) + mv^2/(2*scv^2);
    mess{pt} = mess{pt} - offset;
end

belief_nose = squeeze(score_part{5});

parts = {'left eye','right eye','left mouth','right mouth','nose'};
figure,
for pt = [1:4],
    subplot(2,2,pt);
    imshow(mess{pt},[-2,2]); title(['\mu_{',parts{pt},'-> nose}(X)'],'fontsize',20);
    belief_nose = belief_nose + mess{pt};
end

figure,
subplot(1,2,1);
imshow(input_image);
subplot(1,2,2);
imagesc(max(belief_nose,-10));
axis image;

%% Root-to-leaves message passing

mess_to_leaves = cell(1,4);
for pt = [1:4]
    sch = sg{pt}(1);
    scv = sg{pt}(2);
    mh  = mn{pt}(1);
    mv  = mn{pt}(2);
    
    def(1) = 1/(2*sch^2);
    def(2) = -2*mh/(2*sch^2);
    def(3) = 1/(2*scv^2);
    def(4) = -2*mv/(2*scv^2);
    
    msg_sum = zeros(sv,sh);
    for i = 1:4
        if i~=pt
            msg_sum = msg_sum + mess{i};
        end
    end
    
    [mess_to_leaves{pt},ix_leaves{pt},iy_leaves{pt}] = dt(squeeze(score_part{5}+msg_sum),def(1),def(2),def(3),def(4));
    offset =  mh^2/(2*sch^2) + mv^2/(2*scv^2);
    mess_to_leaves{pt} = mess_to_leaves{pt} - offset;
end

figure,
for pt = [1:4],
    subplot(2,2,pt);
    imshow(mess_to_leaves{pt},[max(mess_to_leaves{pt}(:))-5,max(mess_to_leaves{pt}(:))]); title(['\mu_{nose->',parts{pt},'}(X)'],'fontsize',20);
end

%% show ground-truth bounding box. 
%% You will need to adapt this code to make it show your bounding box proposals
addpath('util/');
[input_image,points] = load_im(im_id,1,1);

figure,
min_x = min(points(1,:));
max_x = max(points(1,:));
min_y = min(points(2,:));
max_y = max(points(2,:));
score = 1;
bbox  = [min_x,min_y,max_x,max_y,score];
showboxes(input_image,bbox);

%% Home-made box
[input_image,points] = load_im(im_id,1,1);

points = zeros(2,4);
for pt = 1:4
    [max_val, max_idx] = max(mess_to_leaves{pt}(:));
    points(1,pt) = ceil(max_idx/size(mess_to_leaves{pt},1));
    points(2,pt) = mod(max_idx,size(mess_to_leaves{pt},1));
end

figure,
min_x = min(points(1,:));
max_x = max(points(1,:));
min_y = min(points(2,:));
max_y = max(points(2,:));
score = 1;
bbox  = [min_x,min_y,max_x,max_y,score];
showboxes(input_image,bbox);

figure,
imshow(input_image), hold on
scatter(points(1,:), points(2,:),'r','LineWidth',1.5)
