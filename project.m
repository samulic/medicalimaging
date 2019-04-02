% Requires load_nii function from third party 
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

%% Create features for each nifti image of both type of lesion
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
    quantAlgo = 'Equal'; % 'Uniform' ?
    Ng = 8; % number of gray levels

    [ROIonly,levels] = prepareVolume(img, mask, 'PETscan', pixelW, sliceS,...
        1, 5, textType, quantAlgo, Ng);

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
% SVM
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
%% TODO
X = cell2mat(struct2cell(X_df));

fit_svm = fitcsvm(X, Y)
cv_svm  = crossval(fit_svm)

kfoldLoss(cv_svm)
