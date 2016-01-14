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
SNR=0;
U=U+SNR*randn(size(U));
V=V+SNR*randn(size(U));
U=max(min(U,1),0);
V=max(min(V,1),0);
figure
nsamps=16;
samp_dim=ceil(sqrt(nsamps));
i=1;
for fcnum=floor(linspace(1,nimages,nsamps));
    subplot(samp_dim,samp_dim,i)
    imshow(cat(2,U(:,:,fcnum),V(:,:,fcnum)))
    axis equal tight
    i=i+1;
end

%% Preprocess Images
% Now, we must preprocess the images using the |apply_image_functions| 
% function before registration and ordering, to
% remove any imaging and/or experimental artifacts.

% number of pixels
% images will be reduced to npixels x npixels resolution
npixels = nres;

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
U = apply_image_functions(U, npixels, dim, channel_weight, ...
    channel_blur, channel_normalize, channel_mean_center, resize_image);
V = apply_image_functions(V, npixels, dim, channel_weight, ...
    channel_blur, channel_normalize, channel_mean_center, resize_image);

% % plot the images (optional)
% figure
% nsamps=16;
% samp_dim=ceil(sqrt(nsamps));
% i=1;
% for fcnum=floor(linspace(1,nimages,nsamps));
%     subplot(samp_dim,samp_dim,i)
%     imshow(cat(2,U(:,:,fcnum),V(:,:,fcnum)))
%     axis equal tight
%     i=i+1;
% end
X=cat(2,U,V);
%% Convert
total_snaps=nimages;
% Select training data
tr_picks=1:3:total_snaps;
tr_snaps=length(tr_picks);
te_picks=sort([2:3:total_snaps 3:3:total_snaps]);
te_snaps=length(te_picks);
% Run POD on training data
R=zeros(tr_snaps);
for i=1:tr_snaps;
    for j=i:tr_snaps;
        R(i,j)=1/tr_snaps*sum(sum(X(:,:,tr_picks(i)).*X(:,:,tr_picks(j))));
        R(j,i)=R(i,j);
    end
end
[vecs,vals]=eig(R);
vals=diag(vals);
[vals,ind]=sort(vals,'descend');
vecs=vecs(:,ind);
max_kept=10;%------------------------------------------------------max_kept
disp(cumsum(vals(1:max_kept))/sum(vals));
pods=zeros(nres,2*nres,max_kept);
for j=1:max_kept;
    for i=1:tr_snaps
        pods(:,:,j)=pods(:,:,j)+vecs(i,j)*X(:,:,i);
    end
%     pods(j,:)=smooth(pods(j,:),7);
end
%% 
k_RMS=zeros(1,max_kept);
u_RMS=zeros(1,max_kept);
t_RMS=zeros(1,max_kept);
figure(4)
hold all
known=1:nres;
unknown=(nres+1):(2*nres);
for n_kept=1:max_kept
    % Test
    M=zeros(n_kept);
    for i=1:n_kept
        for j=i:n_kept
            M(i,j)=sum(sum(pods(:,known,i).*pods(:,known,j)));
            M(j,i)=M(i,j);
        end
    end
    g=zeros(nres,2*nres,total_snaps);
    b=zeros(total_snaps,n_kept);
    for i=1:total_snaps;
        f=zeros(n_kept,1);
        for j=1:n_kept;
            f(j)=sum(sum(X(:,known,i).*pods(:,known,j)));
        end
        b(i,:)=(M\f);
        for j=1:n_kept;
            g(:,:,i)=g(:,:,i)+b(i,j)*pods(:,:,j);
        end
        %g(i,:)=smooth(g(i,:),9);
    end
    g=max(min(g,1),0);
    disp(n_kept)
    k_RMS(n_kept)=sqrt(mean(mean(mean((X(:,known,te_picks)-g(:,known,te_picks)).^2))));
    disp(k_RMS(n_kept))
    u_RMS(n_kept)=sqrt(mean(mean(mean((X(:,unknown,te_picks)-g(:,unknown,te_picks)).^2))));
    disp(u_RMS(n_kept))
    % Plot
    if n_kept==1 || k_RMS(n_kept)<best_RMS
        best_RMS=k_RMS(n_kept);
        best_kept=n_kept;
        figure(2)
        nsamps=te_snaps;
        samp_dim=ceil(sqrt(nsamps));
        for i=1:te_snaps;
            subplot(samp_dim,samp_dim,i)
            imshow(cat(2,X(:,unknown,te_picks(i)),g(:,unknown,te_picks(i))))
            axis equal tight
        end
    end
    figure(4)
    beta=mvregress(b(tr_picks,:),tr_picks');
    te_times=b(te_picks,:)*beta;
    plot(te_picks,te_times,'-')
    t_RMS(n_kept)=sqrt(mean((te_times-te_picks').^2));
end
plot(1:total_snaps,'-k','LineWidth',2)
legend(num2str(t_RMS'))
figure(3)
hold all
plot(k_RMS)
plot(u_RMS)
legend('Error in known','Error in unknown')
figure(4)
title('Times Predicted by POD')
xlabel('Actual')
ylabel('Predicted')