function [features,labels,posterior] = construct_data(nsamples,dataset,problem,plusminus)
switch dataset
    case 'train'
        %% training points are randomly generated (sampled)
        %% this ensures we'll get the same samples
        randn('seed',0); rand('seed',0);
    case 'test'
        randn('seed',500); rand('seed',500);
end

if strcmp(problem,'linear')
    distribution_1  = construct_distribution('gaussian_21_1');
    distribution_2 = construct_distribution('mixture_1');
else
    distribution_1  = construct_distribution('mixture_2');
    distribution_2  = construct_distribution('mixture_3');
end
samples_1       = sample_distribution(distribution_1,nsamples/2);
samples_2       = sample_distribution(distribution_2,nsamples/2);
likelihood_1    = evaluate_distribution_on_grid([0:.02:1],[0:.02:1],distribution_1);
likelihood_2    = evaluate_distribution_on_grid([0:.02:1],[0:.02:1],distribution_2);
posterior       = 1./(1+ (likelihood_2./likelihood_1));


%% gather all 
features = [samples_1,samples_2];
labels   = [ones(1,size(samples_1,2)),zeros(1,size(samples_2,2))];

idxs = randperm(size(features,2));
features = features(:,idxs);
labels   = labels(idxs);

features(3,:) = 1; 

if nargin==4
    if strcmp(plusminus,'plusminus')
        labels = 2*labels - 1;
        features = features([1,2],:);
    end
end


function distr = construct_distribution(distr_name)
switch distr_name,
    case 'gaussian_11_1',
        %% mean
        mn = [5;4.2];

        %% covariance  = u*s*u'
        %% s: lengths of principal axes
        %% u: rotation matrix
        s = diag([2,.1]);
        th_1 = pi/4;
        u = [cos(th_1),-sin(th_1);sin(th_1),cos(th_1)];
        cov = u*s*u';
       
    case 'gaussian_11_2',
        mn = [4.2;5];
        %% covariance
        th_2 = pi/8;
        s = diag([3,.1]);
        u = [cos(th_2),-sin(th_2);sin(th_2),cos(th_2)];
        cov = u*s*u';
    case {'gaussian_12_1','gaussian_12_2'},
        mn = [0;0];
        s = diag([2,.1]);
        th = 0;
        u = [cos(th),-sin(th);sin(th),cos(th)];
        cov  = u*s*u';

    case 'gaussian_12_2',
        mn = [0;0];
        s = diag([2,.1]);
        th = pi/2;
        u = [cos(th),-sin(th);sin(th),cos(th)];
        cov  = u*s*u';
    case 'gaussian_21_1',
        s =.2*diag([.1,.01]);
        th = -pi/8;
        u = [cos(th),-sin(th);sin(th),cos(th)];
        cov  = u*s*u';
        mn  = [.6;.6];
    case 'gaussian_21_2',
        s = .2*diag([.1,.01]);
        th = -pi/8;
        u = [cos(th),-sin(th);sin(th),cos(th)];
        cov  = u*s*u';
        mn  = [.5;.5];
    case 'mixture_1',
        s = .2*diag([.1,.01]);
        th = -pi/8;
        u = [cos(th),-sin(th);sin(th),cos(th)];
        cov{1}  = u*s*u';
        mn{1} =  [.5;.5];
        cov{2} = cov{1};
        mn{2} =[.5;.1];
        prior = [.8,.2];
    case 'mixture_2',
        v = [.1,.1,.1].^2;
        mn =[.4,.4,.6;.8,.3,.4];
        priors = ones(1,3)/3;
    case 'mixture_3'
        v = [.1,.1,.1,.1].^2;
        mn =[.8,.8,.2,.4;.2,.5,.2,.6];
        priors = ones(1,4)/4;
end

if exist('priors','var'),
    for k = 1:length(priors),
        distr(k).mn = [mn(:,k)];
        distr(k).cov  = v(k)*eye(size(mn,1));
        distr(k).prior = priors(k);
    end
else
    if iscell(cov),
        for k=1:length(cov),
            distr(k).mn = mn{k};
            distr(k).cov = cov{k};
            distr(k).prior = prior(k);
        end
    else
        distr(1).mn = mn;
        distr(1).cov  = cov;
        distr(1).prior = 1;
    end
end


function samples_agg = sample_distribution(distr,nsamples);
priors = [distr.prior];
cum_priors = cumsum(priors);

rand_01 = rand(1,nsamples);
samples_agg = [];
for it = 1:length(priors),
    wt = find(rand_01<=cum_priors(it));
    rand_01(wt) = inf;
    samples = sample_gaussian(distr(it),length(wt));
    samples_agg = [samples_agg,samples];
end


function [res,grid_x,grid_y] = evaluate_distribution_on_grid(loc_x,loc_y,distr);
[grid_x,grid_y] = meshgrid([loc_x],[loc_y]);
[sz_m,sz_n] = size(grid_x);
coordinates = [grid_x(:)';grid_y(:)'];

res = 0;
for it = 1:length(distr),
    if isfield(distr,'prior'),
        prior = distr(it).prior;
    else
        prior = 1;
    end
    val = evaluate_gaussian(coordinates,distr(it));
    res = res + val.*prior;
end

res =  reshape(res,[sz_m,sz_n]);

function M = sample_gaussian(gauss_distr, N)

% SAMPLE_GAUSSIAN Draw N random row vectors from a Gaussian distribution
% samples = sample_gaussian(gauss_distr, N)
% adapted from  KEVIN MURPHY's toolbox 

% If Y = CX, Var(Y) = C Var(X) C'.
% So if Var(X)=I, and we want Var(Y)=Sigma, we need to find C. s.t. Sigma = C C'.
% Since Sigma is psd, we have Sigma = U D U' = (U D^0.5) (D'^0.5 U').

mn = gauss_distr.mn;
n =  size(mn,1);
[U,D,V] = svd(gauss_distr.cov);
M = randn(n,N);
M = (U*sqrt(D))*M + mn*ones(1,N); 




function res = evaluate_gaussian(data,distribution);
%% res = evaluate_gaussian(data,distribution);
%%
%% data: (dimension_data) X (number_data) array
%% distribution: structure with fields mn (mean) and cov (covariance)
%% 
%% returns res =  1/((2pi)^(n/2) det(cov)^(1/2)) exp( - 1/2 (x- mn)^T cov^{-1} (x-mn))
%%  
[ndims,ndata]  = size(data);
denominator    =  (2*pi)^(ndims/2)*sqrt(det(distribution.cov));
diff_from_mean = (data-repmat(distribution.mn,[1,ndata]));
precision_matrix =  inv(distribution.cov);

res  = 1/(denominator)*exp(-sum(diff_from_mean.*(precision_matrix*(diff_from_mean)),1)/2);
