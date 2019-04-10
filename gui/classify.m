function yhat = classify(nii, beta)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
stacknii(1) = nii
xtrain = calc_features(stacknii)
% Preprocess input to correct format
X_table = struct2table(xtrain)
X = X_table{:,:} %create matrix for X
%X = normc(X)
yhat_prob = glmval(beta, X, 'logit')
yhat = (yhat_prob >= 0.4)
end

