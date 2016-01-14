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
SNR=0.0;
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
% Select training data
tr_picks=4:5:total_snaps;
tr_snaps=length(tr_picks);
te_picks=sort([1:5:total_snaps 2:5:total_snaps 3:5:total_snaps 5:5:total_snaps]);
te_snaps=length(te_picks);
% Run PCA on training data
max_kept=5;%------------------------------------------------------max_kept
[pcs,score,latent]=pca(X(tr_picks,:),'NumComponents',max_kept,'Centered',false);
pcs=pcs';
k_RMS=zeros(1,max_kept);
u_RMS=zeros(1,max_kept);
t_RMS=zeros(1,max_kept);
figure(4)
hold all
res=size(X,2);
known=1:res/2;
unknown=(res/2+1:res);
for n_kept=1:max_kept
    % Test
    M=zeros(n_kept);
    for i=1:n_kept
        for j=i:n_kept
            M(i,j)=sum(pcs(i,known).*pcs(j,known));
            M(j,i)=M(i,j);
        end
    end
    g=zeros(total_snaps,res);
    b=zeros(total_snaps,n_kept);
    for i=1:total_snaps;
        f=zeros(n_kept,1);
        for j=1:n_kept;
            f(j)=sum(X(i,known).*pcs(j,known));
        end
        b(i,:)=(M\f)';
        for j=1:n_kept;
            g(i,:)=g(i,:)+b(i,j)*pcs(j,:);
        end
        %g(i,:)=smooth(g(i,:),9);
    end
    disp(n_kept)
    k_RMS(n_kept)=sqrt(mean(mean((X(te_picks,known)-g(te_picks,known)).^2)));
    disp(k_RMS(n_kept))
    u_RMS(n_kept)=sqrt(mean(mean((X(te_picks,unknown)-g(te_picks,unknown)).^2)));
    disp(u_RMS(n_kept))
    % Plot
    if n_kept==1 || k_RMS(n_kept)<best_RMS
        best_RMS=k_RMS(n_kept);
        best_kept=n_kept;
        figure(2)
        subplot(1,3,1)
        imagesc(X(te_picks,unknown)+X_mean(te_picks,unknown))
        colormap(jet)
        colorbar
        subplot(1,3,2)
        imagesc(g(:,unknown)+X_mean(:,unknown))
        colorbar
        subplot(1,3,3)
        imagesc(abs(X(te_picks,unknown)-g(te_picks,unknown)))
        colorbar
        figure(5)
        subplot(1,3,1)
        imagesc(X(te_picks,known)+X_mean(te_picks,known))
        colormap(jet)
        colorbar
        subplot(1,3,2)
        imagesc(g(:,known)+X_mean(:,known))
        colorbar
        subplot(1,3,3)
        imagesc(abs(X(te_picks,known)-g(te_picks,known)))
        colorbar
    end
    figure(4)
    beta=mvregress([ones(tr_snaps,1),b(tr_picks,:)],tr_picks');
    te_times=[ones(te_snaps,1),b(te_picks,:)]*beta;
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

figure(2)
subplot(1,3,1)
title('Channel 2: Actual')
subplot(1,3,2)
title('Channel 2: Gappy POD')
subplot(1,3,3)
title('Absolute Prediction Error')