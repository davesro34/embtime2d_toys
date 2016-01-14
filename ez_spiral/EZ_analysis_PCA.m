close all
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
rng(1)
SNR=0.4;
U=U+SNR*randn(size(U));
V=V+SNR*randn(size(U));
% U=max(min(U,1),0);
% V=max(min(V,1),0);


[R,W]=compute_pairwise_alignments(U,10);
[v,d]=eig(R);
[d,ind]=sort(diag(d),'descend');
v=v(:,ind);
R_opt=v(:,1:2);
U=register_all_images(U,R_opt);
V=register_all_images(V,R_opt);
X=cat(2,U,V);

X=reshape(X,2*nres^2,nimages)';
total_snaps=nimages;
X_mean=X;
for i=1:total_snaps
    X_mean(i,:)=mean(X);
end
X=X-X_mean;
%% 

% Select "training" data
tr_picks=4:5:total_snaps;
tr_snaps=length(tr_picks);
tr_movie=X(tr_picks,:);
te_picks=1:total_snaps;
te_picks=te_picks(~ismember(1:total_snaps,tr_picks));
te_snaps=length(te_picks);

% Run PCA on training data
max_kept=1;%------------------------------------------------------max_kept
[pcs,score,latent]=pca(X(tr_picks,:),'NumComponents',max_kept,'Centered',false);
pcs=pcs';
k_RMS=zeros(1,max_kept);
u_RMS=zeros(1,max_kept);
t_RMS=zeros(1,max_kept);
figure(2)
subplot(1,2,1)
hold all
plot(tr_picks,score(:,1),'ok');
title('a)')
xlabel('Actual Time')
ylabel('First Principal Component')
subplot(1,2,2)
hold all
plot([0 total_snaps],[0 total_snaps],'-k')
for n_kept=1:max_kept
    % Test
    M=zeros(n_kept);
    for i=1:n_kept
        for j=i:n_kept
            M(i,j)=sum(pcs(i,:).*pcs(j,:));
            M(j,i)=M(i,j);
        end
    end
    g=zeros(size(X));
    b=zeros(total_snaps,n_kept);
    for i=1:total_snaps;
        f=zeros(n_kept,1);
        for j=1:n_kept;
            f(j)=sum(X(i,:).*pcs(j,:));
        end
        b(i,:)=(M\f)';
        for j=1:n_kept;
            g(i,:)=g(i,:)+b(i,j)*pcs(j,:);
        end
    end
    te_times=interp1(score(:,1),tr_picks',b(te_picks,1),'linear','extrap');
    plot(te_picks,te_times,'.')
    t_RMS(n_kept)=sqrt(mean((te_times-te_picks').^2));
end
for i=1:total_snaps
    blah(i,1)=sum(X(i,:).*pcs(1,:));
end
subplot(1,2,1)
plot(te_picks,b(te_picks,1),'.r')
legend('Training Points','Testing Points')
subplot(1,2,2)
te_picks=1:total_snaps;
te_picks=te_picks(~ismember(1:total_snaps,tr_picks));
plot(te_picks,te_times,'.r')
title('b)')
legend('Reference','Testing Points')
xlabel('Actual Time')
ylabel('Interpolated Time')
RMSE=sqrt(mean((te_times'-te_picks).^2));
disp(RMSE)