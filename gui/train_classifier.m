function [fit, coef, accuracy] = train_classifier(Y, X, classifier, alpha)
%TRAIN a model specified in classifier variable
%
% INPUT
% Y is the target binary vector of lesion type
% X is the design matrix of all features
% Classifier is a string of the classifier to train (lasso, en, svm)
% Optional alpha value for lasso or elastic net
%
% OUTPUT
% fit is the train model fit
%

% Preprocess input to correct format
X_table = struct2table(X);
X = X_table{:,:}; %create matrix for X
Y = Y(:); %create vector for Y
% Normalize data with mean zero and sd 1
X = normc(X);

%train e test
cc = cvpartition(Y,'HoldOut',0.3);
idxTrain = training(cc,1); 
idxTest = ~idxTrain;
XTrain = X(idxTrain,:);
yTrain = Y(idxTrain);
XTest = X(idxTest,:);
yTest = Y(idxTest);

% fixed folds for different models using CV 
c = cvpartition(size(XTrain, 1), 'KFold', 10);

if strcmp(classifier, "lasso") || strcmp(classifier, "elasticnet")
    if strcmp(classifier, "lasso")
        alpha = 1;
    end
    [fit, coef, accuracy] = elasticnet(XTrain, yTrain, XTest, yTest, alpha);
elseif strcmp(classifier, "svm")
    % TODO Perform PCA first
    % DO SVM
    opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
        'AcquisitionFunctionName','expected-improvement-plus'); 
    fit = fitcsvm(XTrain, yTrain,'KernelFunction','rbf',...
        'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);

    [svmyhat,~] = predict(fit,XTest);

    accuracy = sum(svmyhat == yTest) / size(yTest,1);
    cm = confusionmat(yTest, svmyhat)
    coef = NaN;
else
    msg = ["Don't know how to deal with classifier of type " classifier];
    error(msg)
end

