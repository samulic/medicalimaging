

path_o = '../data/lesions/homogeneous/';
path_e = '../data/lesions/heterogeneous/';
files = dir(fullfile([path_o '*.nii']));
features = zeros(size(files, 1), 5);
for n = 1:size(files,1)
    pathtemp = [path_o files(n).name];
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
    features(n, 1) = calc_vol(img, hdr); 
end
    
