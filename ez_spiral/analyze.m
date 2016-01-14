close all
figure
nplots=80;
HL=10;
nres=201;
data=zeros(40401,2,nplots);
h=waitbar(0,'Loading data');
for fcnum=1:nplots;
    waitbar(fcnum/nplots,h);
    fid = fopen(sprintf('outputs/fc%d.dat',fcnum),'r');
    datacell = textscan(fid, '%f%f', 'HeaderLines', HL, 'CollectOutput', 1);
    fclose(fid);
    data(:,:,fcnum) = datacell{1};
end
close(h);
U=reshape(data(:,1,:),nres,nres,nplots);
V=reshape(data(:,2,:),nres,nres,nplots);
variables=['u';'v'];
% darken corners
center=ceil(nres/2);
r_sq=(center-1)^2;
for i=1:nres;
    for j=1:nres;
        dist_sq=(i-center)^2+(j-center)^2;
        if (dist_sq>r_sq)
            U(i,j,:)=.8;
            V(i,j,:)=.8;
        end
    end
end