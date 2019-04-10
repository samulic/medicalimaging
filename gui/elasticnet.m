function [fit, coef, cm] = elasticnet(XTrain, YTrain, XTest, YTest, cv_part, alpha)
%ELASTICNET
%
% INPUT
% Standardized (normalized) X design matrix of training
% Y binary outcome vector (homogeneous = 0 / heterogeneous = 1)
% Alpha optional parameter for the percentage of mix of ridge and lasso
% cv_part partitions for CV (used to compare different models)
%
% OUTPUT
% fit of whole model with all lambdas
% coef lambda1se coefficients
% confusion matrix on the test set

if nargin == 5
    alpha = 1; % lasso
elseif nargin == 4
    alpha = 1;
    cv_part = 10; % 10 fold CV
end

[coef_all, fit] = lassoglm(XTrain, YTrain, 'binomial', ...
    'NumLambda', 100, 'CV', cv_part, 'Alpha', alpha);
indx = fit.Index1SE; %coefficients of lambda1se
coef = coef_all(:, indx);

intercept = fit.Intercept(indx);
coef = [intercept; coef];

%test evaluation
yhat = glmval(coef, XTest, 'logit');
% trasforma in 0 o 1 la probabilita' prevista (yhat)
yhatBool = (yhat >= 0.5);
yTestBool = (YTest == 1); % cambia in tipo booleano
%0 = homogeneous; 1 = heterogeneous
cm = confusionmat(yTestBool, yhatBool);
%labels = ["homogeneous"; "heterogeneous"];
%c3 = confusionchart(c2, labels);
%accuracy = (cm(1,1) + cm(2,2)) / sum(sum(cm));
end

