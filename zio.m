clear; clc; close all;
addpath(genpath('data'));
addpath(genpath('thirdparty-libraries'));

% Create a log file
diary('log.txt');
diary on


%-------------------------------------------------------------------------
% 2. Working with NIfTI images
%-------------------------------------------------------------------------

%-------------------------------------------------------------------------
% 2.1 Loading and visualizing a single NIfTI image
%-------------------------------------------------------------------------
% Remove unnecessary variables before implementing the next task, but
% keep/preserve the variables of interest (if present)
% Tip: in order to preserve some variables of interest, use
% "clearvars -except name-of-the-variable"
close all
clear

gm_o = load_nii('../data/lesions/homogeneous/PHANMAN_130416_2_30_38_m35_t42_OTseg__mod.nii')
gm_e = load_nii('../data/lesions/heterogeneous/PHANMAN_210416_71_79_m75_t39_OTseg__mod.nii')


% Read the header of the loaded image
% Default omogeneous
hdr = gm_o.hdr;
hdr.hk
hdr.dime
hdr.hist

hdr_e = gm_e.hdr;
hdr_e.hk
hdr_e.dime
hdr_e.hist

% Read and visualize the image
% Tip: use the function "squeeze"
gm = gm_o.img;
gm = double(gm);

selectedslice = squeeze(gm(:,:,3));

% "Fix" the visualization based on the output/color scale
minimum = min(min(selectedslice)); disp(num2str(minimum));
maximum = max(max(selectedslice)); disp(num2str(maximum));

% Normalize the image between 0 and 1
selectedslice = selectedslice - minimum;
disp(num2str(max(max(selectedslice))));
selectedslice = selectedslice / maximum;
selectedslice = selectedslice * 1;

% Create a new figure
figuragm = figure();

% Plot the image on the current figure
imshow(selectedslice,[]);

% Visualize the colorbar
colorbar

% Sum all not null pixels
npix = sum(gm ~= 0, 'all')

% Get voxel dimension (in cm)
x = hdr_e.dime.pixdim(2) / 10;
y = hdr_e.dime.pixdim(3) / 10;
z = hdr_e.dime.pixdim(4) / 10;

% Calculate Metabolic Target Volume (in cc)
vol = x * y * z * npix;

