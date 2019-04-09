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
cc = cvpartition(Y, 'HoldOut', 0.3);
idxTrain = training(cc, 1); 
idxTest = ~idxTrain;
XTrain = X(idxTrain,:);
yTrain = Y(idxTrain);
XTest = X(idxTest,:);
yTest = Y(idxTest);


c = cvpartition(size(XTrain, 1), 'KFold', 10);
opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
    'AcquisitionFunctionName','expected-improvement-plus'); 
svmmod = fitcsvm(XTrain, yTrain,'KernelFunction','rbf',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);

[svmyhat,~] = predict(svmmod,XTest);

sum(svmyhat == yTest) / size(yTest,1) %accuracy
confusionmat(yTest, svmyhat)

alpha = 1;
[betas, lassomod] = lassoglm(XTrain, yTrain, 'binomial', 'NumLambda', 100, ...
    'CV', c, 'Alpha', alpha, 'MaxIter', 1e5);
indx = lassomod.Index1SE; %coefficients of lambda1se
coef = betas(:, indx);

intercept = lassomod.Intercept(indx);
coef = [intercept; coef];
sum(coef ~= 0)
%test evaluation
lassoyhat = glmval(coef, XTest, 'logit');
% trasforma in 0 o 1 la probabilita' prevista (yhat)
yhatBool = (lassoyhat >= 0.5);
yTestBool = (yTest == 1); % cambia in tipo booleano
%0 = homogeneous; 1 = heterogeneous
cm = confusionmat(yTestBool, yhatBool)
