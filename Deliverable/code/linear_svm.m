function [w, model] = linear_svm(features, labels, cost)

addpath('libsvm/');
% for linear kernel, t = 0
parameter_string = sprintf('-s 0 -t 0 -c %.5f', cost);
model = svmtrain_libsvm(labels, features, parameter_string);

alpha = model.sv_coef; % support vector coefficients
SVs = model.SVs; % features of the associated support vectors
w = alpha'*SVs; % w = \sum_i a_i F_i
%remove rho (constant term) from the weight of the constant component.
w(end) = w(end) - model.rho;