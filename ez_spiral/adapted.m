clear all
close all

%% Basic Algorithm
% The basic steps are as follows:
%
% # Read in images
% # Preprocess the images (smooth and/or equalize channels, mean-center, etc.)
% # Run vector diffusion maps to register and order preprocessed images

%% Read Images
% The first step is to read in images.
% directory where images are stored
image_dir = 'outputs';
% prefix for each image
image_name = 'fc';
% image type/extension
image_ext = 'dat';
% number of images
nimages = 80;
% dimension of images (dim=2 indicates standard 2D images, rather than
% z-stacks)
dim = 2;
% number of header lines in the data file
HL=10;
% number of variables (columns in the data files)
nchannels=2;
nres=201;
data=zeros(nres^2,nchannels,nimages);
h=waitbar(0,'Loading data');
for fcnum=1:nimages;
    waitbar(fcnum/nimages,h);
    fid = fopen(sprintf('%s/%s%d.%s',image_dir,image_name,fcnum,image_ext),'r');
    datacell = textscan(fid, '%f%f', 'HeaderLines', HL, 'CollectOutput', 1);
    fclose(fid);
    data(:,:,fcnum) = datacell{1};
end
close(h);
U=reshape(data(:,1,:),nres,nres,nimages);
V=reshape(data(:,2,:),nres,nres,nimages);
% darken corners
center=ceil(nres/2);
r_sq=(center-1)^2;
for i=1:nres;
    for j=1:nres;
        dist_sq=(i-center)^2+(j-center)^2;
        if (dist_sq>r_sq)
            U(i,j,:)=0;
            V(i,j,:)=0;
        end
    end
end
%% Show images
% An optional step is to now show the images.
rng(100)
U=U+0.0*randn(size(U));
U=max(min(U,1),0);
figure
nsamps=16;
samp_dim=ceil(sqrt(nsamps));
i=1;
for fcnum=floor(linspace(1,nimages,nsamps));
    subplot(samp_dim,samp_dim,i)
    imshow(U(:,:,fcnum))
    axis equal tight
    colorbar
    i=i+1;
end

%% Preprocess Images
% Now, we must preprocess the images using the |apply_image_functions| 
% function before registration and ordering, to
% remove any imaging and/or experimental artifacts.

% number of pixels
% images will be reduced to npixels x npixels resolution
npixels = 201;

% channel weights
% we scale the first (red) channel by half, and keep the second (green) and
% third (blue) channels at their input values
channel_weight = 1;

% channel blur
% we blur each of the channels by 5%
channel_blur = 0;

% channel normalization
% we normalize the first (red) channel using histogram equalization
% we do not normalize the second (green) or third (blue) channels
channel_normalize = 0;

% mean-center
% we use the first (red) channel to detect the edges of the object in order
% to mean center the object
channel_mean_center = 0;

% resize
% we choose to resize the images so all objects are (approximately) 
% the same size, to remove any variations due to size effects 
resize_image = false;

% we then apply these image functions of normalization, blurring,
% reweighting, and mean-centering
images = apply_image_functions(U, npixels, dim, channel_weight, ...
    channel_blur, channel_normalize, channel_mean_center, resize_image);

% plot the images (optional)
figure
nsamps=16;
samp_dim=ceil(sqrt(nsamps));
i=1;
for fcnum=floor(linspace(1,nimages,nsamps));
    subplot(samp_dim,samp_dim,i)
    imshow(images(:,:,fcnum))
    axis equal tight
    colorbar
    i=i+1;
end
%% Calculate pairwise alignments
% We now need to calculate the angles needed to align _pairs_ of images

% angular discretization when computing pairwise aligments
% this means we search for pairwise aligmemnts over 10 degree increments
ang_dis = 10;

% compute the pairwise alignments 
% images are the preprocessed images
% ang_dis is the angular discretization
% R and W store the pairwise alignments and distances, respectively, for
% vector diffusion maps
[R, W] = compute_pairwise_alignments(images, ang_dis);

%% Apply vector diffusion maps
% First pick out training images
tr_nimages=30;
picks=round(linspace(1,nimages,tr_nimages));
tr_images=images(:,:,picks);
tr_W=W(picks,picks);
R_picks=zeros(1,2*tr_nimages);
for i=1:tr_nimages;
    R_picks(2*i-1)=2*picks(i)-1;
    R_picks(2*i)=2*picks(i);
end
tr_R=R(R_picks,R_picks);

% We can now use vector diffusion maps to register and order the images. 

% ncomps is the number of components to compute
% we only compute 1 coordinate because we only need to order the images
% (i.e., sort by the first coordinate)
ncomps = 2;

% epsilon scale for diffusion maps kernel
% eps_scale = 0.25 means that the epsilon in the diffusion maps kernel is
% 1/4 of the median of the pairwise distances between data points
eps_scale = 0.15;

% vector diffusion maps calculates optimal rotations and embedding
% coordinate
[R_opt, embed_coord, D2, tr_eps] = vdm(tr_R, tr_W, eps_scale, ncomps);

% register images
tr_images_registered = register_all_images(tr_images, R_opt);

% order registered images by embedding coordinate
tr_images_analyzed = order_all_images(tr_images_registered, embed_coord);

% plot the images (optional)
plot_images(tr_images_analyzed, dim)

figure
hold all
plot(picks,embed_coord)
plot(picks,embed_coord,'.k')
te_coords=zeros(nimages,ncomps);
te_TIMES=zeros(nimages,1);
count=1;
for i=1:nimages
    if (~ismember(i,picks))
        W2=exp(-W(i,picks).^2/(0.04*median(W(i,picks)))^2);
        W2=W2/sum(W2);
        for j=1:ncomps
            te_coords(i,:)=(W2*embed_coord);%./(D2');
        end
        te_TIMES(i)=W2*(picks');
        plot(i,te_coords(i,:),'.r')
    else
        te_coords(i,:)=embed_coord(count,:);
        te_TIMES(i)=picks(count);
        plot(i,te_coords(i,:),'ok')
        count=count+1;
    end
end
te_times=interp1(embed_coord(:,1),picks,te_coords(:,1),'linear','extrap');
RMSE=sqrt(mean((te_times-(1:nimages)').^2));
disp(RMSE)
figure
hold all
plot(te_times,'.r')
%plot(te_TIMES,'.b')
plot(1:nimages,'-k')
figure
plot(embed_coord(:,1),embed_coord(:,2))
hold all
plot(embed_coord(:,1),embed_coord(:,2),'.k')
scatter(te_coords(:,1),te_coords(:,2),100,(1:nimages)')
colorbar