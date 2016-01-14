% Generate some fake data that kinda looks like fly embryos
close all
thetas=linspace(0,2*pi,17);
times=(.5:.5:20)';
m=length(times);
xs=zeros(m,17);
ys=zeros(m,17);
for i=1:m
    xs(i,:)=cos(thetas);
    ys(i,:)=sin(thetas);
end
% Right side caves in
xs(:,1)=cos(thetas(1))-.8*(1-exp(-times/5));
ys(:,1)=sin(thetas(1));
xs(:,17)=cos(thetas(1))-.8*(1-exp(-times/5));
ys(:,17)=sin(thetas(1));
% Top and bottom oscillate
ys(:,5)=sin(thetas(5))-.1+.2*sin(2*pi*times/max(times));
ys(:,13)=sin(thetas(13))+.1-.2*sin(2*pi*times/max(times));
% Left caves in starting halfway through
xs(m/2:end,8)=cos(thetas(8))+.2*(1-exp(-(times(m/2:end)-times(m/2))/5));
xs(m/2:end,9)=cos(thetas(9))+.3*(1-exp(-(times(m/2:end)-times(m/2))/5));
xs(m/2:end,10)=cos(thetas(10))+.2*(1-exp(-(times(m/2:end)-times(m/2))/5));
% Right cavity widens starting halfway through
xs(m/4:end,2)=cos(thetas(2))-.6*(1-exp(-(times(m/4:end)-times(m/4))/15));
xs(m/4:end,16)=cos(thetas(16))-.6*(1-exp(-(times(m/4:end)-times(m/4))/15));
xs(m/4:end,3)=cos(thetas(3))+.1*(1-exp(-(times(m/4:end)-times(m/4))/15));
xs(m/4:end,15)=cos(thetas(15))+.1*(1-exp(-(times(m/4:end)-times(m/4))/15));
ys(m/4:end,3)=sin(thetas(3))-.1*(1-exp(-(times(m/4:end)-times(m/4))/15));
ys(m/4:end,15)=sin(thetas(15))+.1*(1-exp(-(times(m/4:end)-times(m/4))/15));
fig=figure(1);
whitebg(gcf,'k')
fig.InvertHardcopy='off';
images=uint8(zeros(300,300,3,m));
for i=1:m
    plot(xs(i,:),ys(i,:),'w','LineWidth',15)
    axis([-2 2 -2 2])
    axis equal
    im=frame2im(getframe(gcf));
    images(:,:,1,i)=im(61:360,131:430,1);
    plot(xs(i,2:3),ys(i,2:3),'w','LineWidth',15)
    hold on
    plot(xs(i,7:8),ys(i,7:8),'w','LineWidth',15)
    plot(xs(i,10:11),ys(i,10:11),'w','LineWidth',15)
    plot(xs(i,15:16),ys(i,15:16),'w','LineWidth',15)
    axis([-2 2 -2 2])
    axis equal
    im=frame2im(getframe(gcf));
    images(:,:,2,i)=im(61:360,131:430,1);
    hold off
end
npixels = 100;
channel_weight = [1 1 1];
% channel blur
% we blur each of the channels by 5%
channel_blur = [0.05 0.05 0.05];
% channel normalization
% we normalize the first (red) channel using histogram equalization
% we do not normalize the second (green) or third (blue) channels
channel_normalize = [1 0 0];
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
plot_images(images,2)


W=pdist2(reshape(double(images), [], m)', reshape(double(images), [], m)');
sigma=median(W(:))/5;
A=exp(-W.^2/sigma^2);
for i=1:m
    A(i,:)=A(i,:)/sum(A(i,:));
end
[V,D]=eig(A);
[d,ind]=sort(diag(D),'descend');
V=V(:,ind);
figure
plot(times,V(:,2))
xlabel('Time')
ylabel('dmaps 1')