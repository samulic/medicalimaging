path_o = '../data/lesions/homogeneous/';
path_e = '../data/lesions/heterogeneous/';

files_o = dir(fullfile([path_o '*.nii']));
files_e = dir(fullfile([path_e '*.nii']));

features_o = zeros(size(files_o, 1), 5);
features_e = zeros(size(files_e, 1), 5);

for n = 1:size(files_o,1)
    pathtemp = [path_o files_o(n).name];
    nii = load_nii(pathtemp);
    % Header
    hdr = nii.hdr;

    img = double(nii.img);

    % "Fix" the visualization based on the output/color scale
    minimum = min(min(min(img)));% disp(num2str(minimum));
    maximum = max(max(max(img)));% disp(num2str(maximum));
    
    % Normalize the image between 0 and 1
    img = img - minimum;
    img = img / maximum;
    img = img * 1;
    
    % Calculate Metabolic Target Volume (in cc)
    features_o(n, 1) = calc_vol(img, hdr);
    
    % Calculate surface
    [features_o(n, 2), ~] = calc_surface(img, hdr);
    
    % Calculate spherical disproportion
    radius = (features_o(n, 1) * 3/4 / pi) ^ (1 / 3);
    features_o(n, 3) = features_o(n, 2) / (radius ^ 2 * 4 * pi);
    
    % Calculate sphericity
    features_o(n, 4) = 1 / features_o(n, 3);
    
    % Calculate ratio of surface and volume
    features_o(n, 5) = features_o(n, 2) / features_o(n, 1);
    
    features_fo_o(n) = firstorder__features(img);
end


% features_e - Structural features
% features_fo_e - First order features
% features_glmc_e - Texture features 
for n = 1:size(files_e,1)
    pathtemp = [path_e files_e(n).name];
    nii = load_nii(pathtemp);
    % Header
    hdr = nii.hdr;

    img = double(nii.img);

    % "Fix" the visualization based on the output/color scale
    minimum = min(min(min(img)));% disp(num2str(minimum));
    maximum = max(max(max(img)));% disp(num2str(maximum));
    
    % Normalize the image between 0 and 1
    img = img - minimum;
    img = img / maximum;
    img = img * 1;
    
    %% Structural features
    % Calculate Metabolic Target Volume (in cc)
    features_e(n).mtv = calc_vol(img, hdr);
    % Calculate surface
    [features_e(n).surf, ~] = calc_surface(img, hdr);
    % Calculate spherical disproportion
    radius = (features_e(n).mtv * 3/4 / pi) ^ (1 / 3);
    features_e(n).radius = features_e(n).surf / (radius ^ 2 * 4 * pi);
    % Calculate sphericity
    features_e(n).sphericity = 1 / features_e(n).radius;    
    % Calculate ratio of surface and volume
    features_e(n).surfToVolRatio = features_e(n).surf / features_e(n).mtv;
    
    %% First order features
    features_fo_e(n) = firstorder__features(img);
    
    %% Texture features
    % Prepare input volume for co-occurrence matrix
    mask = img ~= 0; 
    pixelW = hdr.dime.pixdim(2);
    sliceS = hdr.dime.pixdim(4); % slice spacing z-dim
    textType = 'Matrix';
    quantAlgo = 'Equal'; % 'Uniform' ?
    Ng = 8; % number of gray levels

    [ROIonly,levels] = prepareVolume(img, mask, 'PETscan', pixelW, sliceS,...
        1, 5, 'Matrix', 'Equal', Ng);

    glmc = getGLCM(ROIonly, levels);

    features_glmc_e(n) = getGLCMtextures(glmc);
    
    %feature(n) = [features_e, features_glmc_e];

end

% Histogram features
lin_img = reshape(img, [1, size(img, 1) * size(img, 2) * size(img, 3)]);
vol_img = nonzeros(lin_img);


hist(vol_img, 256)

% call to firstorder__features(vol_img);
features = firstorder__features(img);


%% TODO
% Prepare input volume for co-occurrence matrix

mask = img ~= 0; 
pixelW = header.dime.pixdim(2);
sliceS = header.dime.pixdim(4); % slice spacing z-dim
R = 1;
scale = 1;
textType = 'Matrix';
quantAlgo = 'Equal'; %'Uniform' ?
Ng = 8; % number of gray levels

[ROIonly,levels] = prepareVolume(img, mask, 'PETscan', 2.734375, 3.27, 1, 5, 'Matrix', 'Equal', 8)

glmc = getGLCM(ROIonly, levels);

glmc__features = getGLCMtextures(glmc);

%% CLASSIFICATION
% SVM

n_o = size(features_o, 1); % Number of homogeneous tumor patients (obs)
n_e = size(features_e, 1); % Number of heterogeneous tumor patients (obs)

X = [features_o; features_e];

Y = [repmat(0, n_o, 1); repmat(1, n_e, 1)]


fit_svm = fitcsvm(X, Y)
cv_svm  = crossval(fit_svm)

kfoldLoss(cv_svm)

%%