function pca_features = get_pca_features(features, ncomp)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%PCA

[coeff, score, latent, ~, explained] = pca(features);
pca_features = score;

if nargin < 2
    
    %selection of PC for train e test
    ncomp = 0;
    expldev = 0;
    for i=1:size(explained,1)
        ncomp = ncomp+1;
        expldev = expldev + sum(explained(i));
        if expldev > 0.95 * 100
            break
        end
    end
end
pca_features = pca_features(:,1:ncomp);
end

