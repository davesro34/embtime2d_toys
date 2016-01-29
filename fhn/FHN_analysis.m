% Load and plot the data
close all
filename='FHN_1e0';
%gunzip(strcat(filename,'.gz'))
X=importdata(filename);
[total_snaps,~]=size(X);
X=X(4:total_snaps,:);
[total_snaps,res]=size(X);
res=res/2;

% Rescale the data to give both channels equal weight
V=X(:,1:res);
W=X(:,res+1:res*2);
V=V/range(V(:));
W=W/range(W(:));
% Add noise
rng(0);
SNR=0.4;
V=V+SNR*normrnd(0,1,total_snaps,res);
W=W+SNR*normrnd(0,1,total_snaps,res);
X=[V W];
% Plot
figure(1)
subplot(1,2,1)
imagesc(V)
colormap(jet)
caxis([-0.5 0.5])
colorbar
title('Channel 1')
xlabel('Position Index')
ylabel('Time Index')
subplot(1,2,2)
imagesc(W)
caxis([-0.5 0.5])
colorbar
title('Channel 2')
xlabel('Position Index')
ylabel('Time Index')
% Select "training" data
tr_picks=4:5:total_snaps;
tr_snaps=length(tr_picks);
tr_movie=[V(tr_picks,:) W(tr_picks,:)];

% Construct distance and weight matrices
tr_dists=zeros(tr_snaps);
for i=1:tr_snaps;
    for j=i+1:tr_snaps;
        tr_dists(i,j)=sqrt(sum((tr_movie(i,:)-tr_movie(j,:)).^2));
        tr_dists(j,i)=tr_dists(i,j);
    end
end
tr_eps=0.2*median(tr_dists(:)); % Set kernel scale
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
        te_weights=exp(-te_dists.^2/(tr_eps/2)^2);
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
% RMSE=sqrt(mean((te_times(te_picks(te_picks<161))-te_picks(te_picks<161)).^2));
% disp(RMSE)

% figure(3)
% hold all
% for i=(tr_picks-1);
%     plot(all_te_weights(i,:))
% end
% %% 
% figure
% hold all
% krig_coord=vecs(:,2)';
% krig_poly=polyfit(krig_coord,tr_picks,1);
% krig_times=tr_picks-polyval(krig_poly,krig_coord);
% pairs=tr_snaps*(tr_snaps-1)/2;
% krig_space=zeros(1,pairs);
% krig_vario=zeros(1,pairs);
% count=0;
% for i=1:tr_snaps
%     for j=(i+1):tr_snaps
%         count=count+1;
%         krig_space(count)=abs((krig_coord(i)-krig_coord(j)));
%         krig_vario(count)=0.5*(krig_times(i)-krig_times(j))^2;
%         %plot(krig_space(count),krig_vario(count),'.k')
%     end
% end
% [krig_space,krig_ind]=sort(krig_space);
% krig_vario=krig_vario(krig_ind);
% bins=100;
% div=ceil(linspace(1,pairs+1,bins+1));
% small_space=zeros(1,bins);
% small_vario=zeros(1,bins);
% for i=1:bins;
%     small_space(i)=mean(krig_space(div(i):(div(i+1)-1)));
%     small_vario(i)=mean(krig_vario(div(i):(div(i+1)-1)));
%     %plot(small_space(i),small_vario(i),'or')
% end
% beta=1.5;
% alpha=sum((krig_space.^beta).*krig_vario)/sum(krig_space.^(2*beta));
% alpha=alpha*1;
% %plot(krig_space,alpha*krig_space.^beta);
% %%
% krig_Y=[krig_times'; 0];
% krig_V=ones(tr_snaps+1);
% krig_V(tr_snaps+1,tr_snaps+1)=0;
% for i=1:tr_snaps
%     krig_V(i,i)=0;
%     for j=(i+1):tr_snaps
%         krig_V(i,j)=alpha*(abs((krig_coord(i)-krig_coord(j)))^beta);
%         krig_V(j,i)=alpha*(abs((krig_coord(i)-krig_coord(j)))^beta);
%     end
% end
% % --------------------------------------------------------MEASUREMENT ERROR
% krig_E=zeros(tr_snaps+1);
% for i=1:(tr_snaps);
%     krig_E(i,i)=4;
% end
% krig_VinvY=(krig_V-krig_E)\krig_Y;
% xx=linspace(min(krig_coord),max(krig_coord),100000);
% yy=zeros(1,length(xx));
% for i=1:length(xx)
%     krig_V_star=ones(tr_snaps+1,1);
%     for j=1:tr_snaps;
%         krig_V_star(j)=alpha*abs(xx(i)-krig_coord(j))^beta;
%     end
%     yy(i)=dot(krig_V_star,krig_VinvY);
% end
% figure(2)
% subplot(1,2,1);
% hold all
% plot(yy+polyval(krig_poly,xx),xx,'-b')
% %% 
% % interpolate some actual snapshots
% te_timesK=te_times;
% for te_ind=1:total_snaps;
%     if (~ismember(te_ind,tr_picks))
%         krig_V_star=ones(tr_snaps+1,1);
%         for j=1:tr_snaps;
%             krig_V_star(j)=alpha*abs(te_coords(te_ind)-krig_coord(j))^beta;
%         end
%         te_time=dot(krig_V_star,krig_VinvY)+polyval(krig_poly,te_coords(te_ind));
%         figure(2)
%         subplot(1,2,2)
%         hold all
%         plot(te_ind,te_time,'.b')
%         pause(0.00001)
%         te_timesK(te_ind)=te_time;
%     else
%         te_timesK(te_ind)=te_ind;
%     end
% end
% plot([0 total_snaps],[0 total_snaps],'-k')
% title('Predicted Time vs. Actual Time')
% te_picks=1:total_snaps;
% te_picks=te_picks(~ismember(1:total_snaps,tr_picks));
% RMSE=sqrt(mean((te_timesK(te_picks)-te_picks).^2));
% disp(RMSE)