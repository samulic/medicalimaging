function features = firstorder__features(nii_file)
    %
    % INPUT
    % nii_file is the nii image file of segmented 3D volume of the lesion
    % as outputed from load_nii function
    % 
    % OUTPUT
    % features is a matlab structure containing all the first order
    % histogram-based features of the lesion extracted in the code
    %
    
    img = nii_file.img;
    %hdr = nii_file.hdr; % unused
    
    features = [];
    
    % Some preprocessing needed here
    x = size(img, 1);
    y = size(img, 2);
    z = size(img, 3);
    % Linearize image to a row vector of intensities
    img__lin = reshape(img, [1,  x * y * z]);
    % Exclude null intensities
    tumour__volume = nonzeros(img__lin);
    
    [count, ~] = hist(tumour__volume, 256); % The second argument represent the number of bins
    tot = sum(count);
    frequency = count/tot;

    tumour__dimension = size(tumour__volume);
    squared__matrix = tumour__volume.^2;
    
    % Maximum
    tumour__max = max(tumour__volume);
    features.max = tumour__max;
    
    % Minimum
    tumour__min  = min(tumour__volume);
    features.min = tumour__min;
    
    % Mean
    tumour__mean  = mean(tumour__volume);
    features.mean = tumour__mean;
    
    % Median
    tumour__median  = median(tumour__volume);
    features.median = tumour__median;

    % Mean Absolute Deviation (MAD)
    m__dev = abs(tumour__volume - tumour__mean);
    MAD = mean(m__dev);
    features.mad = MAD;
    
    % Root Mean Square (RMS)
    tumour__rms = rms(tumour__volume);
    features.rms = tumour__rms;
    
    % Energy
    tumour__energy = sum(sum(squared__matrix));
    features.energy = tumour__energy;

    
    % Entropy
    entropy__vector = frequency.*log2(frequency);
    for k=1:256
        if isnan(entropy__vector(1, k))
            entropy__vector(1, k) = 0;
        end
    end
    entropy = (-1) * sum(entropy__vector);
    features.entropy = entropy;

    % Kurtosis
    num_vector = (tumour__volume - tumour__mean).^4;
    den_vector = (tumour__volume - tumour__mean).^2;
    num = sum(num_vector)/tumour__dimension(1,1);
    den = ((sum(den_vector) / tumour__dimension(1,1)))^2;
    kurtosis = num / den;
    features.kurtosis = kurtosis;

    % Skewness
    temp__vector = (tumour__volume - tumour__mean).^3;
    num1 = sum(temp__vector)/tumour__dimension(1,1);
    den1 = (sqrt(sum(den_vector)/tumour__dimension(1,1)))^3;
    skewness = num1/den1;
    features.skewness = skewness;
    clear temp__vector
    
    % Standard Deviation 
    tumour__sd = std(tumour__volume);
    features.sd = tumour__sd;

    % Uniformity
    uniformity__vector = frequency.*frequency;
    uniformity = sum(sum(uniformity__vector));
    features.uniformity = uniformity;

    % Variance
    temp__vector = (tumour__volume - tumour__mean).^2;
    variance = (1/(tumour__dimension(1,1)-1)) * sum(sum(temp__vector));
    features.variance = variance;
    clear temp__vector
end