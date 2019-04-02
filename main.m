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
end

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
    
    % Calculate Metabolic Target Volume (in cc)
    features_e(n, 1) = calc_vol(img, hdr);
    
    % Calculate surface
    [features_e(n, 2), ~] = calc_surface(img, hdr);
    
    % Calculate spherical disproportion
    radius = (features_e(n, 1) * 3/4 / pi) ^ (1 / 3);
    features_e(n, 3) = features_e(n, 2) / (radius ^ 2 * 4 * pi);
    
    % Calculate sphericity
    features_e(n, 4) = 1 / features_e(n, 3);
    
    % Calculate ratio of surface and volume
    features_e(n, 5) = features_e(n, 2) / features_e(n, 1);
end

% Histogram features
lin_img = reshape(img, [1, size(img, 1) * size(img, 2) * size(img, 3)]);
vol_img = nonzeros(lin_img);

% call to firstorder__features(vol_img);

[count,~] = hist(vol_img, 256);
hist(vol_img, 128)

tot = sum(count);
frequency = count / tot;



hist(tot, 256)

help hist

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