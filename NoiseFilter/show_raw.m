w=512
h=512;
fid=fopen('output.raw','rb');
GY=fread(fid,[w*h],'uint8');
m=reshape(GY,w,h);
m=m';
m1=m./255;
figure(1)
imshow(m1);
