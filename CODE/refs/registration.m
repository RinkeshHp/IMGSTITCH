i1 = imread("/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/36.50_13.50_3.53.png");
i2 = imread("/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/36.70_13.50_3.53.png");
c = normxcorr2(i1(:,:,1),i2(:,:,1));
figure
surf(c) 
shading flat
[max_c,imax] = max(abs(c(:)));
[ypeak,xpeak] = ind2sub(size(c),imax(1));
corr_offset = [(xpeak-size(i1,2)) 
               (ypeak-size(i1,1))];

xoffset = corr_offset(1);
yoffset = corr_offset(2);


xbegin = round(xoffset + 1);
xend   = round(xoffset + size(i1,2));
ybegin = round(yoffset + 1);
yend   = round(yoffset + size(i1,1));
extracted_i1 = i2(ybegin:yend-1,xbegin:xend-1,:);

recovered_i1 = uint8(zeros(size(i2)));
recovered_i1(ybegin:yend,xbegin:xend,:) = i1;
imshow(recovered_i1)

imshowpair(i2(:,:,1),recovered_i1,"blend")