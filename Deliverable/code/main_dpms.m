%% Maximum Likelihood parameter estimation for pairwise terms
for im_id=1:1000,
    [input_image,points] = load_im(im_id,1,1);
    center = points(:,5);
    offsets(:,:,im_id) = points(:,1:4) - center*ones(1,4);
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

%% compute unary terms
t= load('svm_linear');
for part = 1:5
    weights_unary(part,:) = t.svm_linear{part}.weight;
end

im_id         = 1;
[input_image] = load_im(im_id,1,1);
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

belief_nose=  squeeze(score_part{5});

parts = {'left eye','right eye','left mouth','right mouth','nose'};
for pt = [1:4],
    figure(pt),
    subplot(1,2,1);
    imshow(squeeze(score_part{pt}),[-2,2]); title(['\Phi_{',parts{pt},'}(X)'],'fontsize',20);
    subplot(1,2,2);
    imshow(mess{pt},[-2,2]); title(['\mu_{',parts{pt},'-> nose}(X)'],'fontsize',20);
    belief_nose= belief_nose + mess{pt};
end

figure(5),
subplot(1,2,1);
imshow(input_image);
subplot(1,2,2);
imagesc(max(belief_nose,-10));
axis image;

%% Home-made max-product algorithm

my_mess = cell(1,4);

for pt = [1:4]
    sch = sg{pt}(1);
    scv = sg{pt}(2);
    mh  = -mn{pt}(1);
    mv  = -mn{pt}(2);
    
    my_mess{pt} = zeros(sv,sh);
    for Xr_1 = 1:sv
        for Xr_2 = 1:sh
           to_max = zeros(sv,sh);
            for Xp_1 = 1:sv
                for Xp_2 = 1:sh
                    to_max(Xp_1, Xp_2) = score_part{pt}(Xp_1,Xp_2)...
                        *pairwise(Xp_1, Xp_2, Xr_1, Xr_2, sch, scv, mh, mv);
                end
            end
            my_mess{pt}(Xr_1, Xr_2) = max(to_max(:));
        end
    end
end



%% show ground-truth bounding box. 
%% You will need to adapt this code to make it show your bounding box proposals
[input_image,points] = load_im(im_id,1,1);

figure(6);
min_x = min(points(1,:));
max_x = max(points(1,:));
min_y = min(points(2,:));
max_y = max(points(2,:));
score = 1;
bbox  = [min_x,min_y,max_x,max_y,score];
showboxes(input_image,bbox);





    


