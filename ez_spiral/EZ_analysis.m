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

% Select "training" data
tr_picks=4:5:total_snaps;
tr_snaps=length(tr_picks);
tr_movie=X(tr_picks,:);

% Construct distance and weight matrices
tr_dists=zeros(tr_snaps);
for i=1:tr_snaps;
    for j=i+1:tr_snaps;
        tr_dists(i,j)=sqrt(sum((tr_movie(i,:)-tr_movie(j,:)).^2));
        tr_dists(j,i)=tr_dists(i,j);
    end
end
tr_eps=0.25*median(tr_dists(:)); % Set kernel scale
tr_weights=exp(-(tr_dists.^2)/tr_eps^2);
% Make row stochastic
for i=1:tr_snaps
    norm=sum(tr_weights(i,:));
    tr_weights(i,:)=tr_weights(i,:)/norm;
end
% Find and sort eigenvalues
[vecs,vals]=eig(tr_weights);
vals=diag(vals);
[vals,ind]=sort(vals,'descend');
vecs=vecs(:,ind);
% check (plot) training stuff
figure(2)
subplot(1,2,1)
hold all
plot(tr_picks,vecs(:,2),'ok');
disp(vals(2));
title('a)')
xlabel('Actual Time')
ylabel('Diffusion Maps Coordinate')
% insert other coordinates
te_snaps=total_snaps;
[vec_sorted,vec_ind]=sort(vecs(:,2));
disp(sum(vec_sorted(1:tr_snaps-1)<vec_sorted(2:tr_snaps)))
% average indistinct sites
vec_for_interp=vec_sorted;
times_for_interp=tr_picks(vec_ind);
i=1; % index to check
k=0; % number of times this index has already been averaged
problems=0; % number of pairs that had to be averaged
while(i<length(vec_for_interp))
    if (vec_for_interp(i)==vec_for_interp(i+1));
        avg=((k+1)*times_for_interp(i)+times_for_interp(i+1))/(k+2);
        times_for_interp=[times_for_interp(1:i-1),avg,times_for_interp(i+2:length(times_for_interp))];
        vec_for_interp=[vec_for_interp(1:i);vec_for_interp(i+2:length(vec_for_interp))];
        problems=problems+1;
    else
        i=i+1;
        k=0;
    end
end
disp(problems)
te_times=zeros(1,total_snaps);
te_coords=zeros(1,total_snaps);
all_te_weights=zeros(total_snaps,tr_snaps);
% plot the interpolation curve
S=linspace(min(vec_for_interp),max(vec_for_interp),100000)';
Y=interp1(vec_for_interp,times_for_interp,S);
%plot(Y,S,'-r')
% interpolate some actual snapshots
for te_ind=1:total_snaps;
    if (~ismember(te_ind,tr_picks))
        te_snap=X(te_ind,:);
        te_dists=zeros(1,tr_snaps);
        for j=1:tr_snaps;
            te_dists(j)=sqrt(sum((te_snap-tr_movie(j,:)).^2));
        end
        te_weights=exp(-te_dists.^2/(tr_eps)^2);
        te_weights=te_weights/sum(te_weights);
        all_te_weights(te_ind,:)=te_weights;
        te_coords(te_ind)=sum(te_weights.*vecs(:,2)')/vals(2);
        subplot(1,2,1)
        plot(te_ind,te_coords(te_ind),'.r')
        pause(0.00001)
        te_time=interp1(vec_for_interp,times_for_interp,te_coords(te_ind),'linear','extrap');
        te_times(te_ind)=te_time;
    else
        te_times(te_ind)=te_ind;
    end
end
legend('Training Points','Testing Points')
subplot(1,2,2)
hold all
plot([0 total_snaps],[0 total_snaps],'-k')
te_picks=1:total_snaps;
te_picks=te_picks(~ismember(1:total_snaps,tr_picks));
plot(te_picks,te_times(te_picks),'.r')
title('b)')
legend('Reference','Testing Points')
xlabel('Actual Time')
ylabel('Interpolated Time')
RMSE=sqrt(mean((te_times(te_picks)-te_picks).^2));
disp(RMSE)