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
rng(0)
SNR=.2;
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

% Select movies data
A_picks=1:5:ceil(2*total_snaps/3);
A_snaps=length(A_picks);
B_picks=floor(1*total_snaps/3):3:total_snaps;
B_snaps=length(B_picks);
co_snaps=A_snaps+B_snaps;
A=X(A_picks,:);
B=X(B_picks,:);
% Plot
figure
subplot(2,1,1)
imagesc(A+X_mean(1:A_snaps,:))
title('Movie A')
xlabel('Position')
ylabel('Time Index')
colormap(jet)
caxis([-0.5 0.5]);
colorbar
subplot(2,1,2)
imagesc(B+X_mean(1:B_snaps,:))
title('Movie B')
xlabel('Position')
ylabel('Time Index')
caxis([-0.5 0.5]);
colorbar
% Calculate distance matrix between all snapshots
D=zeros(co_snaps);
for i=1:co_snaps
    for j=(i+1):co_snaps
        if i<=A_snaps
            if j<=A_snaps
                D(i,j)=sqrt(sum((A(i,:)-A(j,:)).^2));
                D(j,i)=D(i,j);
            else
                D(i,j)=sqrt(sum((A(i,:)-B(j-A_snaps,:)).^2));
                D(j,i)=D(i,j);
            end
        else
            D(i,j)=sqrt(sum((B(i-A_snaps,:)-B(j-A_snaps,:)).^2));
            D(j,i)=D(i,j);
        end
    end
end
% Calculate kernel scales and weight matrix
A_eps=0.25*median(median(D(1:A_snaps,1:A_snaps)));
B_eps=0.25*median(median(D(1+A_snaps:co_snaps,1+A_snaps:co_snaps)));
W=zeros(co_snaps);
W(:,1:A_snaps)=exp(-(D(:,1:A_snaps).^2)/(A_eps)^2);
W(:,1+A_snaps:co_snaps)=exp(-(D(:,1+A_snaps:co_snaps).^2)/(B_eps)^2);
% Make Row Stochastic in each block
for i=1:co_snaps;
    W(i,1:A_snaps)=W(i,1:A_snaps)/sum(W(i,1:A_snaps));
    W(i,1+A_snaps:co_snaps)=W(i,1+A_snaps:co_snaps)/sum(W(i,1+A_snaps:co_snaps));
end
%-------------------------- MESS WITH WHAT YOU THINK YOU KNOW
tau_real=min(B_picks)-1;
B_picks=B_picks-tau_real;
pick_err=5;
B_picks=B_picks+normrnd(0,pick_err,size(B_picks));
%--------------------------
% Run DMAPS and Nystrom to give B times within A
[va,da]=eig(W(1:A_snaps,1:A_snaps));
[da,ind]=sort(diag(da),'descend');
va=va(:,ind);
figure
subplot(2,3,1)
plot(A_picks,va(:,2),'.k');
title('Results of Diffusion Maps on A')
xlabel('Original time')
ylabel('Diffusion maps coordinate')
pause(.01)
B2A_coords=W(1+A_snaps:co_snaps,1:A_snaps)*va(:,2)/da(2);
subplot(2,3,2)
plot(B_picks,B2A_coords,'.k');
title('Results of Nystrom on B into A')
xlabel('Original time')
ylabel('Calculated Coordinate')
pause(.01)
B2A_times=interp1(va(:,2),A_picks,B2A_coords,'linear','extrap');
subplot(2,3,3)
plot(B_picks,B2A_times,'.k');
title('Results of Interpolation of B into A')
xlabel('Original time')
ylabel('Calculated/Guessed Time')
hold all
plot(B_picks,B_picks,'or')
% Assume known times for B must be shifted by tau_B. Find tau to minimize
% penalty between known times and the interpolated times.
best_pen=Inf;
for i=1:B_snaps
    tau_temp=B2A_times(i)-B_picks(i);
    temp_pen=merged_penalty(tau_temp,B_picks',B2A_times);
    if temp_pen<best_pen
        tau_B=tau_temp;
        best_pen=temp_pen;
    end
end
f=@(tau_B)merged_penalty(tau_B,B_picks',B2A_times);
tau_B=fminunc(f,tau_B);
plot(B_picks,B_picks+tau_B,'ob')
legend('Calculated Times','Original Times with no shift','Original times plus calculated shift')
pause(.01)
% Run DMAPS and Nystrom to give A times within B
[vb,db]=eig(W(1+A_snaps:co_snaps,1+A_snaps:co_snaps));
[db,ind]=sort(diag(db),'descend');
vb=vb(:,ind);
subplot(2,3,4)
plot(B_picks,vb(:,2),'.k');
title('Results of Diffusion Maps on B')
xlabel('Original time')
ylabel('Diffusion maps coordinate')
pause(.01)
A2B_coords=W(1:A_snaps,1+A_snaps:co_snaps)*vb(:,2)/db(2);
subplot(2,3,5)
plot(A_picks,A2B_coords,'.k');
title('Results of Nystrom on A into B')
xlabel('Original time')
ylabel('Calculated Coordinate')
pause(.01)
A2B_times=interp1(vb(:,2),B_picks,A2B_coords,'linear','extrap');
subplot(2,3,6)
plot(A_picks,A2B_times,'.k');
title('Results of Interpolation of A into B')
xlabel('Original time')
ylabel('Calculated/Guessed Time')
hold all
plot(A_picks,A_picks,'or')
% Assume real times for A must be shifted by tau_A. Find tau to minimize SSE
% between known times and the interpolated times.
best_pen=Inf;
for i=1:A_snaps
    tau_temp=A2B_times(i)-A_picks(i);
    temp_pen=merged_penalty(tau_temp,A_picks',A2B_times);
    if temp_pen<best_pen
        tau_A=tau_temp;
        best_pen=temp_pen;
    end
end
disp(tau_A)
f=@(tau_A)merged_penalty(tau_A,A_picks',A2B_times);
tau_A=fminunc(f,tau_A);
disp(tau_B)
disp(tau_A)
plot(A_picks,A_picks+tau_A,'ob')
legend('Calculated Times','Original Times with no shift','Original times plus calculated shift')
pause(.01)
tau_est=(tau_B-tau_A)/2;
disp(tau_est-tau_real)
% Plot
figure
final_movie=[A;B];
[~,ind]=sort([A_picks,B_picks+tau_est]);
final_movie=final_movie(ind,:);
imagesc(final_movie+X_mean(1:co_snaps,:))
title('Combined Movies')
xlabel('Position')
ylabel('Time Index')
colormap(jet)
colorbar

% i_best=0;
% diff=0;
% for i=floor(B_snaps/10):ceil(B_snaps*.9)
%     p1=polyfit(B_picks(1:i)',B2A_times(1:i),1);
%     p2=polyfit(B_picks(i+1:B_snaps)',B2A_times(i+1:B_snaps),1);
%     tmp_diff=abs(p1(1))-abs(p2(1));
%     if tmp_diff>diff
%         i_best=i;
%         diff=tmp_diff;
%     end
% end
% disp(B_picks(i_best))