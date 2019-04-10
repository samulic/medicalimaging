% Requires load_nii function from third party 
clear; clc; close all;
addpath(genpath('thirdparty-libraries'));

path = 'lesions/';

files_o = dir(fullfile([path 'homogeneous/*.nii']));
files_e = dir(fullfile([path 'heterogeneous/*.nii']));
files = [files_o; files_e];

% Create 'dataframe', first column is id (names of file images)
df = struct( 'id',  {files.name} );
% Second column is the type of the lesion
type = [repmat('omogen', size(files_o, 1), 1); repmat('eterog', size(files_e, 1), 1)];
temp = num2cell(type);
[df.type] = temp{:};

%% Create features for each nifti image
% features are appended to the dataframe
for i = 1:size(df, 2)
    temp_path = [files(i).folder '\' files(i).name];
    nii_image = load_nii(temp_path);
    
    img = double(nii_image.img);
    header = nii_image.hdr;
    
    % "Fix" the visualization based on the output/color scale
    minimum = min(min(min(img)));% disp(num2str(minimum));
    maximum = max(max(max(img)));% disp(num2str(maximum));
    
    % Normalize the image between 0 and 1
    img = img - minimum;
    img = img / maximum;
    img = img * 1;
    
    nii_image.img = img;
    % ### Structural features ### 
    % Calculate Metabolic Target Volume (in cc)
    df(i).vol = calc_vol(img, header);
    % Calculate surface
    [df(i).surf, ~] = calc_surface(img, header);
    % Calculate spherical disproportion
    radius = (df(i).vol * 3/4 / pi) ^ (1 / 3);
    df(i).sphericDisprop = df(i).surf / (radius ^ 2 * 4 * pi);
    % Calculate sphericity
    df(i).sphericity = 1 / df(i).sphericDisprop;    
    % Calculate ratio of surface and volume
    df(i).surfToVolRatio = df(i).surf / df(i).vol;
    firstorder_features = firstorder__features(img, header);
    f = fieldnames(firstorder_features);
    for j = 1:length(f)
        df(i).(f{j}) = firstorder_features.(f{j});
    end
    
    % ### Texture features ###
    % Prepare input volume for co-occurrence matrix
    mask = img ~= 0; 
    pixelW = header.dime.pixdim(2);
    sliceS = header.dime.pixdim(4); % slice spacing z-dim
    textType = 'Matrix';
    %quantAlgo = 'Uniform'; % 'Equal';
    Ng = 64; % number of gray levels

    [ROIonly,levels] = prepareVolume(img, mask, 'PETscan', pixelW, sliceS,...
        1, sliceS, textType, 'Uniform', Ng);

    glmc = getGLCM(ROIonly, levels);

    texture__features = getGLCMtextures(glmc);
    t = fieldnames(texture__features);
    for k = 1:length(t)
        df(i).(t{k}) = texture__features.(t{k});
    end
end

% Histogram features
%lin_img = reshape(img, [1, size(img, 1) * size(img, 2) * size(img, 3)]);
%vol_img = nonzeros(lin_img);

%hist(vol_img, 256)


%% CLASSIFICATION
% Create target binary variable
% Zero if lesion is homogeneous, one if heterogeneous
Y = [];
for i = 1:size(df, 2)
    if isequal(df(i).type, 'o')
        Y = [Y 0];
    else
        Y = [Y 1];
    end
end

X_df = rmfield(df, {'id', 'type'});
%X = cell2mat(struct2cell(X_df));
X = struct2table(X_df, 'AsArray', true);

%% PENALIZED LOGISTIC REGRESSION
X2mtrx = X{:,:}; %lassoglm requires a matrix for X, not a table
Yvec = Y(:); %lassoglm requires a vector for Y


%train e test
cc = cvpartition(Yvec,'HoldOut',0.3);
idxTrain = training(cc,1); 
idxTest = ~idxTrain;
XTrain = X2mtrx(idxTrain,:);
yTrain = Yvec(idxTrain);
XTest = X2mtrx(idxTest,:);
yTest = Yvec(idxTest);

%%%%%%%%%
% LASSO
%%%%%%%%%
rng('default')%seed
tallrng('default')
[B,FitInfo] = lassoglm(XTrain,yTrain,'binomial','NumLambda',25,'CV',10);

%plot lambda
lassoPlot(B,FitInfo,'PlotType','CV'); %(REVERSE X-AXIS)
legend('show','Location','best')

%plot coefficients
lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log'); %(REVERSE X-AXIS)

indx = FitInfo.Index1SE; %coefficients of lambda1se
B0 = B(:,indx)
nonzeros = sum(B0 ~= 0)%7 selected
selectedVar_lasso = X(:,find(B0)).Properties.VariableNames %selected variables

intercept = FitInfo.Intercept(indx);
coef = [intercept; B0]

%test evaluation
yhat = glmval(coef,XTest,'logit');
yhatBool = (yhat>=0.5); %questo trasforma in 0 o 1, la colonna di quell che c'ï¿½ sopra

yTestBool = (yTest==1);
%c = confusionchart(yTestBool,yhatBool);
c2 = confusionmat(yTestBool, yhatBool);%0 = homogeneous; 1 = heterogeneous
labels = ["homogeneous"; "heterogeneous"];
c3 = confusionchart(c2, labels);

% la matrice c2 è mette nella prima riga e nella prima colonna lo 0
% (omogeneo) e quindi i falsi. Ovvero ha una struttura del tipo
%  0 1
%0 # # 
%1 # #
%
%il confusionchart invece ordina in base all'ordine alfavetico delle
%labels, quindi metterà prima 1, ovvero heterogeneous.
%Tutte le metriche vengono calcolate tramite la matrice c2, ricordarsene la
%struttura!!!

acc_lasso=(c2(1,1)+c2(2,2))/sum(sum(c2));
missclass_lasso=1-acc_lasso;%missclassification rate
specificity_lasso = c2(1,1)/sum(c2(1,:)); %TN rate (homogeneous)
sensitivity_lasso = c2(2,2)/sum(c2(2,:)); %TP rate (heterogeneoous)

%%%%%%%%%%
%ELASTIC NET
%%%%%%%%%%
 
%elastic net con ALPHA=0.5
[B,FitInfo] = lassoglm(XTrain,yTrain,'binomial','NumLambda',25,'CV',10, 'Alpha',0.5);

%plot lambda
lassoPlot(B,FitInfo,'PlotType','CV'); %(REVERSE X-AXIS)
legend('show','Location','best')

%plot coefficients
lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log'); %(REVERSE X-AXIS)

indx = FitInfo.Index1SE; %coefficients of lambda1se
B0 = B(:,indx)
nonzeros = sum(B0 ~= 0)%14 selected
selectedVar_enet05 = X(:,find(B0)).Properties.VariableNames %selected variables

intercept = FitInfo.Intercept(indx);
coef = [intercept; B0]

%test evaluation
yhat = glmval(coef,XTest,'logit');
yhatBool = (yhat>=0.5); %questo trasforma in 0 o 1, la colonna di quell che c'ï¿½ sopra

yTestBool = (yTest==1);
%c = confusionchart(yTestBool,yhatBool);
c2 = confusionmat(yTestBool, yhatBool);%0 = homogeneous; 1 = heterogeneous
labels = ["homogeneous"; "heterogeneous"];
c3 = confusionchart(c2, labels);

acc_enet05=(c2(1,1)+c2(2,2))/sum(sum(c2));
missclass_enet05=1-acc_enet05;%missclassification rate
specificity_enet05 = c2(1,1)/sum(c2(1,:)); %TN rate (homogeneous)
sensitivity_enet05 = c2(2,2)/sum(c2(2,:)); %TP rate (heterogeneoous)

%%%%%%%%%%%%%%%%%%
%elastic net con ALPHA=0.2
[B,FitInfo] = lassoglm(XTrain,yTrain,'binomial','NumLambda',25,'CV',10, 'Alpha',0.2);

%plot lambda
lassoPlot(B,FitInfo,'PlotType','CV'); %(REVERSE X-AXIS)
legend('show','Location','best')

%plot coefficients
lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log'); %(REVERSE X-AXIS)

indx = FitInfo.Index1SE; %coefficients of lambda1se
B0 = B(:,indx)
nonzeros = sum(B0 ~= 0)%18 selected
selectedVar_enet02 = X(:,find(B0)).Properties.VariableNames  %selected variables


intercept = FitInfo.Intercept(indx);
coef = [intercept; B0]

%test evaluation
yhat = glmval(coef,XTest,'logit');
yhatBool = (yhat>=0.5); %questo trasforma in 0 o 1, la colonna di quell che c'ï¿½ sopra

yTestBool = (yTest==1);
%c = confusionchart(yTestBool,yhatBool);
c2 = confusionmat(yTestBool, yhatBool);%0 = homogeneous; 1 = heterogeneous
labels = ["homogeneous"; "heterogeneous"];
c3 = confusionchart(c2, labels);

acc_enet02=(c2(1,1)+c2(2,2))/sum(sum(c2));
missclass_enet02=1-acc_enet02;%missclassification rate
specificity_enet02 = c2(1,1)/sum(c2(1,:)); %TN rate (homogeneous)
sensitivity_enet02 = c2(2,2)/sum(c2(2,:)); %TP rate (heterogeneoous)

%%%%%%%%%%%%%%%%%%
%elastic net con ALPHA=0.8
[B,FitInfo] = lassoglm(XTrain,yTrain,'binomial','NumLambda',25,'CV',10, 'Alpha',0.8);

%plot lambda
lassoPlot(B,FitInfo,'PlotType','CV'); %(REVERSE X-AXIS)
legend('show','Location','best')

%plot coefficients
lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log'); %(REVERSE X-AXIS)

indx = FitInfo.Index1SE; %coefficients of lambda1se
B0 = B(:,indx)
nonzeros = sum(B0 ~= 0)%7 selected
selectedVar_enet08 = X(:,find(B0)).Properties.VariableNames %selected variables

intercept = FitInfo.Intercept(indx);
coef = [intercept; B0]

%test evaluation
yhat = glmval(coef,XTest,'logit');
yhatBool = (yhat>=0.5); %questo trasforma in 0 o 1, la colonna di quell che c'ï¿½ sopra

yTestBool = (yTest==1);
c = confusionchart(yTestBool,yhatBool);
c2 = confusionmat(yTestBool, yhatBool);
labels = ["homogeneous"; "heterogeneous"];
c3 = confusionchart(c2, labels);
 
acc_enet08=(c2(1,1)+c2(2,2))/sum(sum(c2));
missclass_enet08=1-acc_enet08;%missclassification rate
specificity_enet08 = c2(1,1)/sum(c2(1,:)); %TN rate (homogeneous)
sensitivity_enet08 = c2(2,2)/sum(c2(2,:)); %TP rate (heterogeneoous)

%% OSS
% sopra alpha=0.5 non cambia piï¿½ la matrice di confusione, quindi aumentare
% il fattore di penalizzazione alpha non cambia il risultato.

%% PCA + SVM

%PCA
[coeff,score,latent,~,explained] = pca(XTrain);
XpcaTrain = score;

%selection of PC for train e test
ncomp = 0;
expldev = 0;
for i=1:size(explained,1)
    ncomp = ncomp+1;
    expldev = expldev + sum(explained(i));
    if expldev > 95
        break
    end
end
XpcaTrain = XpcaTrain(:,1 :ncomp)



%classification with SVM
c = cvpartition(size(XTrain, 1), 'KFold', 10);
opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
    'AcquisitionFunctionName','expected-improvement-plus'); 
svmmod = fitcsvm(XpcaTrain, yTrain,'KernelFunction','rbf',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);

[coeff,score,latent,~,explained] = pca(XTest);
XpcaTest = score;
XpcaTest = XpcaTest(:,1:ncomp);

[svmyhat,~] = predict(svmmod,XpcaTest);

sum(svmyhat == yTest) / size(yTest,1) %accuracy
confusionmat(yTest, svmyhat)
%% SAVE WORKSPACE
save('workspace')
%% Export to CSV
dataset = struct2table(df, 'AsArray', true);
%csvwrite('target.csv', Y')
writetable(dataset, 'dataset.csv')

B = TreeBagger(100, X, Y);

view(B.Trees{1, 2})
view(B)