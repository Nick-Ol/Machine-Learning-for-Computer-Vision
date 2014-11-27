function  [training_set_features,training_set_targets,validation_set_features,validation_set_targets] = split_data(inputs,outputs,nsamples,K,validation_run);
%%  [training_set_inputs,training_set_targets,validation_set_features,validation_set_targets] = split_data(inputs,outputs,nsamples,K,validation_run);
%% ----------------------------------------------------------
%% Splits data into training set and validation set 
%% ----------------------------------------------------------

%% pick indexes for the data that will be used as validation set
nvalidation_set      = floor(nsamples/K);
rand('seed',0);
permutation = randperm(nsamples);

validation_indexes   = nvalidation_set*(validation_run -1) + [1:nvalidation_set];
training_indexes     = setdiff([1:nsamples],validation_indexes);

validation_indexes = permutation(validation_indexes);
training_indexes   = permutation(training_indexes);


[srt,idx] = sort(outputs(training_indexes),'descend');
training_indexes = training_indexes(idx);

training_set_features = inputs(:,training_indexes);
training_set_targets  = outputs(training_indexes);

[srt,idx] = sort(outputs(validation_indexes),'descend');
validation_indexes = validation_indexes(idx);


validation_set_features = inputs(:,validation_indexes);
validation_set_targets  = outputs(validation_indexes);
