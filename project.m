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

%% SVM senza feature selection
fit_svm = fitcsvm(X, Y);
cv_svm  = crossval(fit_svm);

kfoldLoss(cv_svm)

%% PENALIZED LOGISTIC REGRESSION
X2mtrx = X{:,:}; %lassoglm requires a matrix for X, not a table
Yvec = Y(:); %lassoglm requires a vector for Y


%train e test
c = cvpartition(Yvec,'HoldOut',0.3);
idxTrain = training(c,1); 
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
nonzeros = sum(B0 ~= 0)%3 selected

intercept = FitInfo.Intercept(indx);
coef = [intercept; B0]

%test evaluation
yhat = glmval(coef,XTest,'logit');
yhatBool = (yhat>=0.5); %questo trasforma in 0 o 1, la colonna di quell che c'� sopra

yTestBool = (yTest==1);
c = confusionchart(yTestBool,yhatBool);

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

intercept = FitInfo.Intercept(indx);
coef = [intercept; B0]

%test evaluation
yhat = glmval(coef,XTest,'logit');
yhatBool = (yhat>=0.5); %questo trasforma in 0 o 1, la colonna di quell che c'� sopra

yTestBool = (yTest==1);
c = confusionchart(yTestBool,yhatBool);

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

intercept = FitInfo.Intercept(indx);
coef = [intercept; B0]

%test evaluation
yhat = glmval(coef,XTest,'logit');
yhatBool = (yhat>=0.5); %questo trasforma in 0 o 1, la colonna di quell che c'� sopra

yTestBool = (yTest==1);
c = confusionchart(yTestBool,yhatBool);

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

intercept = FitInfo.Intercept(indx);
coef = [intercept; B0]

%test evaluation
yhat = glmval(coef,XTest,'logit');
yhatBool = (yhat>=0.5); %questo trasforma in 0 o 1, la colonna di quell che c'� sopra

yTestBool = (yTest==1);
c = confusionchart(yTestBool,yhatBool);
c2 = confusionmat(yTestBool, yhatBool);
c3 = confusionchart(c2);

%RECALL
for i =1:size(c2,1)
    recall(i)=c2(i,i)/sum(c2(i,:));
    T = c2(i,i);
    T = c2(i,i);
end
Recall=sum(recall)/size(c2,1);
 

%% OSS
% sopra alpha=0.5 non cambia pi� la matrice di confusione, quindi aumentare
% il fattore di penalizzazione alpha non cambia il risultato.

% Export to CSV
dataset = struct2table(df, 'AsArray', true);
%csvwrite('target.csv', Y')
writetable(dataset, 'dataset.csv')

B = TreeBagger(100, X, Y);

view(B.Trees{1, 2})
view(B)