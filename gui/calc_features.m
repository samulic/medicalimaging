function features = calc_features(stacknii)
    %
    % INPUT
    % stacknii is the stack of nifti images of 3D volume of the lesion 
    % the first dimension is the index of the image, 
    % the second is the full nifti image (as outputed from load_nii)
    %
    % OUTPUT
    % features is a matlab structure containing all the features extracted 
    % in the code
    %
    
    features = struct();
    
    % Total number of lesions
    n = size(stacknii, 1);
    
    %for i = n:-1:1 % start from the last to preallocate space of structure
    % this will mess with the ordered target variable! KISS
    for i = 1:n
        nifti = stacknii(i);
        img_orig = double(nifti.img);
        header = nifti.hdr;

        % "Fix" the visualization based on the output/color scale
        minimum = min(min(min(img_orig)));% disp(num2str(minimum));
        maximum = max(max(max(img_orig)));% disp(num2str(maximum));

        % Normalize the image between 0 and 1
        img = img_orig - minimum;
        img = img / maximum;
        img = img * 1;
        % Use normalized images for calculating the features
        nifti.img = img;
        
        %%%%%% Structural features %%%%%% 
        % Calculate Metabolic Target Volume (in cc)
        features(i).mtv = calc_volume(nifti);
        % Calculate surface
        features(i).surf = calc_surface(nifti);
        % Calculate spherical disproportion
        radius = (features(i).mtv * 3/4 / pi) ^ (1 / 3);
        features(i).sphericDisprop = features(i).surf / (radius ^ 2 * 4 * pi);
        % Calculate sphericity
        features(i).sphericity = 1 / features(i).sphericDisprop;    
        % Calculate ratio of surface and volume
        features(i).surfToVolRatio = features(i).surf / features(i).mtv;

        %%%%%% Histogram-based features %%%%%% 
        fo_features = firstorder__features(nifti);
        f = fieldnames(fo_features);
        % "Concatenate" to struct of all features
        for j = 1:length(f)
            features(i).(f{j}) = fo_features.(f{j});
        end
        
        %%%%%% Texture features %%%%%% 
        % Parameters to prepare volume for co-occurrence matrix
        mask = img ~= 0; 
        scan_type = 'PETscan';
        pixel_width = header.dime.pixdim(2);
        slice_spacing = header.dime.pixdim(4); % slice spacing z-dim
        R = 1; % Ratio of weight to band-pass coefficients over the weigth
                % of the rest of coefficients (HHH and LLL)
                % R=1 to not perform wavelet band-pass filtering
        scale = 'pixelW'; % scale at which 'volume' is isotropically resampled (mm)
                          % 'pixelW' --> resampled at the initial in-plane resolution
        textType = 'Matrix';
        quantAlgo = 'Uniform'; % 'Equal';
        gray_levels = 64; % number of gray levels

        [ROIonly, levels] = prepareVolume(img, mask, scan_type, pixel_width, ...
            slice_spacing, R, scale, textType, quantAlgo, gray_levels);

        glcm = getGLCM(ROIonly, levels);

        texture_features = getGLCMtextures(glcm);
        t = fieldnames(texture_features);
        % "Concatenate" to struct of all features
        for k = 1:length(t)
            features(i).(t{k}) = texture_features.(t{k});
        end
    end
end