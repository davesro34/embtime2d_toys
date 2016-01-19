%%% Generate and color fake fly embryos
%% Set Parameters
rng(0)
white_noise=10; % standard deviation of white noise applied to every pixel
intra_noise=.01; % standard deviation of vertex noise within each movie
inter_noise=.013; % standard deviation of vertex noise between movies
nmovies=3;
nimages=40*nmovies*2;
for i=1:nmovies
    movie_idx{i}=(2*i-1):(2*nmovies):nimages;
end
movie_idx{nmovies+1}=2:2:nimages;
%% Define underlying process (the base path)
close all
npoints=17;
thetas=linspace(0,2*pi,npoints);
times=linspace(2/nimages,20,nimages);
xs=zeros(nimages,17);
ys=zeros(nimages,17);
for i=1:nimages
    xs(i,:)=cos(thetas);
    ys(i,:)=sin(thetas);
end
% Right side caves in
xs(:,1)=cos(thetas(1))-.6*(1-exp(-times/5));
ys(:,1)=sin(thetas(1));
xs(:,17)=cos(thetas(1))-.6*(1-exp(-times/5));
ys(:,17)=sin(thetas(1));
% Top and bottom oscillate
ys(:,5)=sin(thetas(5))-.1+.2*sin(pi*times/max(times));
ys(:,13)=sin(thetas(13))+.1-.2*sin(pi*times/max(times));
% Left caves in starting halfway through
xs(nimages/2:end,8)=cos(thetas(8))+.8*(1-exp(-(times(nimages/2:end)-times(nimages/2))/20));
xs(nimages/2:end,9)=cos(thetas(9))+1*(1-exp(-(times(nimages/2:end)-times(nimages/2))/15));
xs(nimages/2:end,10)=cos(thetas(10))+.8*(1-exp(-(times(nimages/2:end)-times(nimages/2))/20));
% Right cavity widens starting quarter through
xs(nimages/4:end,2)=cos(thetas(2))-.65*(1-exp(-(times(nimages/4:end)-times(nimages/4))/10));
xs(nimages/4:end,16)=cos(thetas(16))-.65*(1-exp(-(times(nimages/4:end)-times(nimages/4))/10));
xs(nimages/4:end,3)=cos(thetas(3))+.3*(1-exp(-(times(nimages/4:end)-times(nimages/4))/15));
xs(nimages/4:end,15)=cos(thetas(15))+.3*(1-exp(-(times(nimages/4:end)-times(nimages/4))/15));
ys(nimages/4:end,3)=sin(thetas(3))-.7*(1-exp(-(times(nimages/4:end)-times(nimages/4))/10));
ys(nimages/4:end,15)=sin(thetas(15))+.7*(1-exp(-(times(nimages/4:end)-times(nimages/4))/10));
xs(3*nimages/4:end,4)=cos(thetas(4))+.4*(1-exp(-(times(3*nimages/4:end)-times(3*nimages/4))/10));
xs(3*nimages/4:end,14)=cos(thetas(14))+.4*(1-exp(-(times(3*nimages/4:end)-times(3*nimages/4))/10));
ys(3*nimages/4:end,4)=sin(thetas(4))-.2*(1-exp(-(times(3*nimages/4:end)-times(3*nimages/4))/10));
ys(3*nimages/4:end,14)=sin(thetas(14))+.2*(1-exp(-(times(3*nimages/4:end)-times(3*nimages/4))/10));
%% Generate data
for i=1:nmovies+1
    movie_noise{i}=i*inter_noise*randn(2,npoints);
end
fig=figure(1);
whitebg(gcf,'k')
fig.InvertHardcopy='off';
images=uint8(zeros(300,300,3,nimages));
for movie=1:nmovies+1
    for i=movie_idx{movie}
        all_noise=movie_noise{movie}+intra_noise*randn(2,npoints);
        if movie==nmovies+1
            all_noise=all_noise+inter_noise*randn(2,npoints);
        end
        xs(i,:)=xs(i,:)+all_noise(1,:);
        ys(i,:)=ys(i,:)+all_noise(2,:);
        t = cumsum(sqrt([0,diff(xs(i,:))].^2 + [0,diff(ys(i,:))].^2));
        tt=linspace(0,max(t),85);
        xx=smooth(interp1(t,xs(i,:),tt),1);
        yy=smooth(interp1(t,ys(i,:),tt),1);
        plot(xx,yy,'w','LineWidth',15)
        axis([-2 2 -2 2])
        axis equal
        limits=axis;
        im=frame2im(getframe(gcf));
        images(:,:,1,i)=.5*im(61:360,131:430,1);
        plot(xs(i,3:4),ys(i,3:4),'w','LineWidth',10)
        hold on
        if i>nimages/2
            intensity=1-exp(-(i-nimages/2)/15);
            plot(xs(i,8:9),ys(i,8:9),'Color',[intensity 0 0],'LineWidth',10)
            plot(xs(i,9:10),ys(i,9:10),'Color',[intensity 0 0],'LineWidth',10)
        end
        plot(xs(i,14:15),ys(i,14:15),'w','LineWidth',10)
        axis(limits)
        im=frame2im(getframe(gcf));
        images(:,:,2,i)=im(61:360,131:430,1);
        hold off
    end
end
npixels = 100;
channel_weight = [1 1 1];
% channel blur
% we blur each of the channels by 5%
channel_blur = [0.04 0.05 0.04];
% channel normalization
% we normalize the first (red) channel using histogram equalization
% we do not normalize the second (green) or third (blue) channels
channel_normalize = [0 0 0];
% mean-center
% we use the first (red) channel to detect the edges of the object in order
% to mean center the object
channel_mean_center = [0 0 0];
% resize
% we choose to resize the images so all objects are (approximately)
% the same size, to remove any variations due to size effects
resize_image = false;
% we then apply these image functions of normalization, blurring,
% reweighting, and mean-centering
images = apply_image_functions(images, npixels, 2, channel_weight, ...
channel_blur, channel_normalize, channel_mean_center, resize_image);
% for i=1:nmovies+1
%     plot_images(images(:,:,:,movie_idx{i}),2)
% end
%%
images=double(images);
noisy=images+white_noise*randn(size(images));
noisy=min(noisy,256); noisy=max(noisy,0);
% plot_images(uint8(noisy),2);
W=pdist2(reshape(double(noisy(1:end,:,1,:)), [], nimages)', reshape(double(noisy(1:end,:,1,:)), [], nimages)');
sigma=median(W(:))/5;
A=exp(-W.^2/sigma^2);
for i=1:nimages
    A(i,:)=A(i,:)/sum(A(i,:));
end
[V,D]=eig(A);
[d,ind]=sort(diag(D),'descend');
V=V(:,ind);
figure
subplot(1,2,1)
hold on
for i=1:nmovies+1
    labels{i}=sprintf('Movie %d',i);
    plot(times(movie_idx{i}),V(movie_idx{i},2),'.')
end
labels{nmovies+1}='Snapshots';
legend(labels)
xlabel('Time')
ylabel('dmaps 1')
subplot(1,2,2)
hold on
for i=1:nmovies+1
    plot(V(movie_idx{i},3),V(movie_idx{i},2),'.')
end
legend(labels)
xlabel('dmaps 2')
ylabel('dmaps 1')
%% Color
train_ind=movie_idx{nmovies+1};
test_ind=1:nimages; test_ind(train_ind)=[];
test_colored=noisy(:,:,:,test_ind);
for i=1:length(test_ind)
    [~,ind]=sort(abs(V(train_ind,2)-V(test_ind(i),2)));
    test_colored(:,:,2,i)=mean(noisy(:,:,2,train_ind(ind(1:5))),4);
end
error=(test_colored(:,:,2,:)-images(:,:,2,test_ind)).^2;
weight=(test_colored(:,:,2,:)+images(:,:,2,test_ind));
disp(sqrt(sum(error(:))/size(images,1)^2/nimages))
% plot_images(uint8(cat(2,test_colored,noisy(:,:,:,test_ind),images(:,:,:,test_ind))),2)
% plot_images(uint8(cat(2,test_colored(:,:,2,:),noisy(:,:,2,test_ind),images(:,:,2,test_ind))),2)