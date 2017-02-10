function [imo,rois] = cnn_wsddn_get_batch(images, imdb, batch, opts)
% cnn_wsddn_get_batch  Load, preprocess, and pack images for CNN evaluation

if isempty(images)
  imo = [] ;
  rois = [] ;
  return ;
end

% fetch is true if images is a list of filenames (instead of
% a cell array of images)
fetch = ischar(images{1}) ;

% prefetch is used to load images in a separate thread
prefetch = fetch & opts.prefetch ;

% pick size
imSize = imdb.images.size(batch(1),:);
factor = min(opts.scale(1)/imSize(1),opts.scale(1)/imSize(2));
height = floor(factor*imSize(1));

if prefetch
  vl_imreadjpeg(images, 'numThreads',opts.numThreads,'Resize',height,'prefetch') ;
  imo = [] ;
  rois = [] ;
  return ;
end

if fetch
  ims = vl_imreadjpeg(images,'numThreads',opts.numThreads,'Resize',height) ;
else
  ims = images ;
end

for i=1:numel(images)
  % acquire image
  if isempty(ims{i})
    imt = imread(images{i}) ;
    if size(imt,3) == 1
      imt = cat(3, imt, imt, imt) ;
    end
    
    ims{i} = imresize(imt,factor,'Method',opts.interpolation);
    ims{i} = single(ims{i}) ; % faster than im2single (and multiplies by 255)
  end
end



bboxes = cell(1,numel(batch));
nBoxes = 0;
for b=1:numel(batch)
  bboxes{b} = double(imdb.images.boxes{batch(b)});
  nBoxes = nBoxes + size(bboxes{b},1);
end
 

rois = zeros(nBoxes,5);
countr = 0;

maxW = 0;
maxH = 0;



for b=1:numel(batch)
  
  hw = imdb.images.size(batch(b),:);
  h = hw(1);
  w = hw(2);
  
  imsz = size(ims{b});
  
  if opts.flip(b)
    im = ims{b};
    ims{b} = im(:,end:-1:1,:);
    
    bbox = bboxes{b};
    bbox(:,[2,4]) = w + 1 - bbox(:,[4,2]);
    bboxes{b} = bbox;
  end
  

  maxH = max(imsz(1),maxH);
  maxW = max(imsz(2),maxW);
 
  % adapt bounding boxes into new coord
  bbox = bboxes{b};
  if any(bbox(:)<=0)
    error('bbox error');
  end
  nB = size(bbox,1);
  tbbox = scale_box(bbox,[h,w],imsz);
  if any(tbbox(:)<=0)
    error('tbbox error');
  end

  rois(countr+1:countr+nB,:) = [b*ones(nB,1),tbbox];
  countr = countr + nB;
end

% rois = single(rois);
depth = size(ims{1},3);
imo = zeros(maxH,maxW,depth,numel(batch),'single');

if isempty(opts.averageImage)
  avgIm = [];
elseif numel(opts.averageImage)==depth
  avgIm = opts.averageImage;
else
  avgIm = single(imssize(opts.averageImage,[maxH,maxW],'Method',opts.interpolation));
end


for b=1:numel(batch)
  sz = size(ims{b});

  imo(1:sz(1),1:sz(2),:,b) = single(ims{b});
  
  if ~isempty(avgIm)
    if numel(opts.averageImage)==size(imo,3)
      imo(1:sz(1),1:sz(2),:,b) = single(bsxfun(@minus,imo(1:sz(1),1:sz(2),:,b),opts.averageImage));
    else
      imo(1:sz(1),1:sz(2),:,b) = imo(1:sz(1),1:sz(2),:,b) - avgIm(1:sz(1),1:sz(2),:);
    end
  
  end

end


function boxOut = scale_box(boxIn,szIn,szOut)
  
  h = szIn(1);
  w = szIn(2);

  bxr = 0.5 * (boxIn(:,2)+boxIn(:,4)) / w;
  byr = 0.5 * (boxIn(:,1)+boxIn(:,3)) / h;
 
  bwr = (boxIn(:,4)-boxIn(:,2)+1) / w;
  bhr = (boxIn(:,3)-boxIn(:,1)+1) / h;
  
  % boxIn center in new coord
  byhat = (szOut(1) * byr);
  bxhat = (szOut(2) * bxr);
  
  % relative width, height
  bhhat = szOut(1) * bhr;
  bwhat = szOut(2) * bwr;
  
  % transformed boxIn
  boxOut = [max(1,round(byhat - 0.5 * bhhat)),...
    max(1,round(bxhat - 0.5 * bwhat)), ...
    min(szOut(1),round(byhat + 0.5 * bhhat)),...
    min(szOut(2),round(bxhat + 0.5 * bwhat))];

