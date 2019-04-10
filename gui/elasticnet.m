function [fit, coef, cm] = elasticnet(XTrain, YTrain, XTest, YTest, alpha)
%ELASTICNET
%
% INPUT
% Standardized (normalized) X design matrix of training
% Y binary outcome vector (homogeneous = 0 / heterogeneous = 1)
%
% OUTPUT
% fit of whole model with all lambdas
% coef lambda1se coefficients
% accuracy of lambda1se model

if nargin == 4
  alpha = 1; % lasso
end

rng('default')%seed
tallrng('default')
[coef_all, fit] = lassoglm(XTrain, YTrain, 'binomial', ...
    'NumLambda', 100, 'CV', 10, 'Alpha', alpha);
indx = fit.Index1SE; %coefficients of lambda1se
coef = coef_all(:, indx);

intercept = fit.Intercept(indx);
coef = [intercept; coef];

%test evaluation
yhat = glmval(coef, XTest, 'logit');
% trasforma in 0 o 1 la probabilita' prevista (yhat)
yhatBool = (yhat >= 0.4);
yTestBool = (YTest == 1); % cambia in tipo booleano
%0 = homogeneous; 1 = heterogeneous
cm = confusionmat(yTestBool, yhatBool);
%labels = ["homogeneous"; "heterogeneous"];
%c3 = confusionchart(c2, labels);
%accuracy = (cm(1,1) + cm(2,2)) / sum(sum(cm));

end

