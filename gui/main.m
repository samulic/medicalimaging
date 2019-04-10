path = '../lesions/';

files_o = dir(fullfile([path 'homogeneous/*.nii']));
files_e = dir(fullfile([path 'heterogeneous/*.nii']));
files = [files_o; files_e];

for i = 1:size(files, 1)
    temp_path = [files(i).folder '\' files(i).name];
    nii_image = load_nii(temp_path);
    
    stacknii(i, :) = nii_image;
end

features = calc_features(stacknii);
X = struct2table(features);
X = X{:,:}; %create matrix for X
Y = [zeros(size(files_o, 1), 1); ones(size(files_e, 1),1)];
Y = Y(:);

% Normalize data with mean zero and sd 1
X = normc(X);

%train e test
rng('default');
tallrng('default');

cc = cvpartition(Y, 'HoldOut', 0.3);
idxTrain = training(cc, 1); 
idxTest = ~idxTrain;
XTrain = X(idxTrain,:);
XTrain_pca = get_pca_features(XTrain); % deviance threshold 0.95
yTrain = Y(idxTrain);
XTest = X(idxTest,:);
XTest_pca = get_pca_features(XTest, size(XTrain_pca, 2));
yTest = Y(idxTest);


c = cvpartition(size(XTrain, 1), 'KFold', 10);
opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
    'AcquisitionFunctionName','expected-improvement-plus'); 
svm_mod = fitcsvm(XTrain, yTrain,'KernelFunction','rbf',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);

[svmyhat,~] = predict(svm_mod,XTest);

sum(svmyhat == yTest) / size(yTest,1) %accuracy
svm_cm = confusionmat(yTest, svmyhat)

[betas, lasso_mod] = lassoglm(XTrain, yTrain, 'binomial', 'NumLambda', 100, ...
    'CV', 10, 'Alpha', 1, 'MaxIter', 1e5);
indx = lasso_mod.Index1SE; %coefficients of lambda1se
coef = betas(:, indx);

intercept = lasso_mod.Intercept(indx);
lasso_coef = [intercept; coef];
sum(lasso_coef ~= 0)
%test evaluation
lassoyhat = glmval(lasso_coef, XTest, 'logit');
% trasforma in 0 o 1 la probabilita' prevista (yhat)
yhatBool = (lassoyhat >= 0.5);
yTestBool = (yTest == 1); % cambia in tipo booleano
%0 = homogeneous; 1 = heterogeneous
lasso_cm = confusionmat(yTestBool, yhatBool)


[betas, en_02_mod] = lassoglm(XTrain, yTrain, 'binomial', 'NumLambda', 100, ...
    'CV', c, 'Alpha', 0.2, 'MaxIter', 1e5);
indx = en_02_mod.Index1SE; %coefficients of lambda1se
coef = betas(:, indx);

intercept = en_02_mod.Intercept(indx);
en_02_coef = [intercept; coef];
sum(en_02_coef ~= 0)
%test evaluation
en_02_yhat = glmval(en_02_coef, XTest, 'logit');
% trasforma in 0 o 1 la probabilita' prevista (yhat)
yhatBool = (en_02_yhat >= 0.5);
yTestBool = (yTest == 1); % cambia in tipo booleano
%0 = homogeneous; 1 = heterogeneous
en_02_cm = confusionmat(yTestBool, yhatBool)


[betas, en_05_mod] = lassoglm(XTrain, yTrain, 'binomial', 'NumLambda', 100, ...
    'CV', c, 'Alpha', 0.5, 'MaxIter', 1e5);
indx = en_05_mod.Index1SE; %coefficients of lambda1se
coef = betas(:, indx);

intercept = en_05_mod.Intercept(indx);
en_05_coef = [intercept; coef];
sum(en_05_coef ~= 0)
%test evaluation
en_05_yhat = glmval(en_05_coef, XTest, 'logit');
% trasforma in 0 o 1 la probabilita' prevista (yhat)
yhatBool = (en_05_yhat >= 0.5);
yTestBool = (yTest == 1); % cambia in tipo booleano
%0 = homogeneous; 1 = heterogeneous
en_05_cm = confusionmat(yTestBool, yhatBool)


[betas, en_08_mod] = lassoglm(XTrain, yTrain, 'binomial', 'NumLambda', 100, ...
    'CV', c, 'Alpha', 0.8, 'MaxIter', 1e5);
indx = en_08_mod.Index1SE; %coefficients of lambda1se
coef = betas(:, indx);

intercept = en_08_mod.Intercept(indx);
en_08_coef = [intercept; coef];
sum(en_08_coef ~= 0)
%test evaluation
en_08_yhat = glmval(en_08_coef, XTest, 'logit');
% trasforma in 0 o 1 la probabilita' prevista (yhat)
yhatBool = (en_08_yhat >= 0.5);
yTestBool = (yTest == 1); % cambia in tipo booleano
%0 = homogeneous; 1 = heterogeneous
en_08_cm = confusionmat(yTestBool, yhatBool)


save("models", "svm_mod", "svm_cm", "lasso_mod", "lasso_coef", "lasso_cm",...
"en_05_mod", "en_05_cm","en_08_mod","en_08_cm")


acc  = (en_02_cm(1,1) + en_02_cm(2,2)) / sum(sum(en_02_cm))
sens = en_02_cm(2,2) / sum(en_02_cm(2,:))
spec = en_02_cm(1,1) / sum(en_02_cm(1,:))


%% Export to CSV
dataset = struct2table(features, 'AsArray', true);
%csvwrite('target.csv', Y')
writetable(dataset, 'features.csv')
