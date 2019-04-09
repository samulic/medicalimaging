function volume = calc_volume(nii_file)
%CALC_VOL
img = nii_file.img;
header = nii_file.hdr;

x = header.dime.pixdim(2) / 10;
y = header.dime.pixdim(3) / 10;
z = header.dime.pixdim(4) / 10;
npix = sum(img ~= 0, 'all');

% Calculate Metabolic Target Volume (in cc)
volume = x * y * z * npix;
end

