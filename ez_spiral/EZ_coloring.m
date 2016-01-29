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
images=cat(3,reshape(data(:,1,:),nres,nres,1,nimages),reshape(data(:,2,:),nres,nres,1,nimages));
images=cat(3,images,zeros(nres,nres,1,nimages));
% darken corners
center=ceil(nres/2);
r_sq=(center-1)^2;
for i=1:nres;
    for j=1:nres;
        dist_sq=(i-center)^2+(j-center)^2;
        if (dist_sq>r_sq)
            images(i,j,:,:)=0;
        end
    end
end
plot_images(images,2)
%%
noise=.2;
noisy=images+noise*randn(size(images));
noisy=min(noisy,1); noisy=max(noisy,0);
plot_images(noisy,2);
W=pdist2(reshape(double(noisy(:,:,1,:)), [], nimages)', reshape(double(noisy(:,:,1,:)), [], nimages)');
sigma=median(W(:))/5;
A=exp(-W.^2/sigma^2);
for i=1:nimages
    A(i,:)=A(i,:)/sum(A(i,:));
end
[V,D]=eig(A);
[d,ind]=sort(diag(D),'descend');
V=V(:,ind);
test_ind=2:2:nimages;
train_ind=1:nimages; train_ind(test_ind)=[];
figure
subplot(1,2,1)
hold on
plot((train_ind),V(train_ind,2),'.')
plot((test_ind),V(test_ind,2),'.')
legend('Training','Testing')
xlabel('Time')
ylabel('dmaps 1')
subplot(1,2,2)
hold on
plot(V(train_ind,3),V(train_ind,2),'.')
plot(V(test_ind,3),V(test_ind,2),'.')
legend('Training','Testing')
xlabel('dmaps 2')
ylabel('dmaps 1')
%% Color
test_colored=noisy(:,:,:,test_ind);
for i=1:length(test_ind)
    [~,ind]=sort(abs(V(train_ind,2)-V(test_ind(i),2)));
    test_colored(:,:,2,i)=mean(noisy(:,:,2,train_ind(ind(1:5))),4);
end
error=(test_colored(:,:,2,:)-images(:,:,2,test_ind)).^2;
disp(sqrt(sum(error(:))/nres^2/nimages))
plot_images(cat(2,test_colored,noisy(:,:,:,test_ind),images(:,:,:,test_ind)),2)
plot_images(cat(2,test_colored(:,:,2,:),noisy(:,:,2,test_ind),images(:,:,2,test_ind)),2)