function [outputArg1] = calc_surf(img, hdr)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
header = hdr;
x = header.dime.pixdim(2) / 10;
y = header.dime.pixdim(3) / 10;
npix = sum(img ~= 0, 'all')

% Calculate Metabolic Target Volume (in cc)
surf = x * y * npix;
outputArg1 = surf;
end

