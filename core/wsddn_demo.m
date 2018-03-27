function wsddn_demo(varargin)
% @author: Hakan Bilen
% cnn_wsddn_demo : this script shows a detection demo


%addpath('layers');
%addpath('pascal');
%addpath(fullfile('layers','matlab'));
%run(fullfile('matconvnet', 'matlab', 'vl_setupnn.m')) ;
%addpath(fullfile('matconvnet','examples'));

opts.dataDir = fullfile('data') ;
opts.expDir = fullfile('exp') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.modelPath = fullfile('models', 'imagenet-vgg-f.mat') ;
opts.proposalDir = fullfile('data','SSW');

% if you have limited gpu memory (<6gb), you can change the next 2 params
opts.maxNumProposals = inf; % limit number
% opts.imageScales = [480,576,688,864,1200]; % scales
opts.imageScales = [480,576,688,864,1200]; % scales

opts.train.gpus = [] ;
opts.train.prefetch = true ;

opts.numFetchThreads = 1 ;
opts = vl_argparse(opts, varargin) ;

display(opts);
if ~exist(fullfile(opts.dataDir,'VOCdevkit','VOCcode','VOCinit.m'),'file')
  error('VOCdevkit is not installed');
end
addpath(fullfile(opts.dataDir,'VOCdevkit','VOCcode'));
opts.train.expDir = opts.expDir ;
% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
net = load(opts.modelPath);
net = dagnn.DagNN.loadobj(net) ;

net.mode = 'test' ;
if ~isempty(opts.train.gpus)
  net.move('gpu') ;
end

if isfield(net,'normalization')
  bopts = net.normalization;
else
  bopts = net.meta.normalization;
end

bopts.interpolation = 'bilinear';
bopts.imageScales = opts.imageScales;
bopts.numThreads = opts.numFetchThreads;
% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------
fprintf('loading imdb...');
if exist(opts.imdbPath,'file')==2
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_voc07_eb_setup_data('dataDir',opts.dataDir, ...
    'proposalDir',opts.proposalDir,'loadTest',1);
    
  save(opts.imdbPath,'-struct', 'imdb', '-v7.3');
end

fprintf('done\n');
minSize = 20;
imdb = fixBBoxes(imdb, minSize, opts.maxNumProposals);

% --------------------------------------------------------------------
%                                                               Detect
% --------------------------------------------------------------------
% query images
testIdx = [12,15];

VOCinit;
cats = VOCopts.classes;
ovTh = 0.4; % nms threshold
scTh = 0.1; % det confidence threshold

bopts.useGpu = numel(opts.train.gpus) >  0 ;

detLayer = find(arrayfun(@(a) strcmp(a.name, 'xTimes'), net.vars)==1);

net.vars(detLayer(1)).precious = 1;
% run detection
rcolors = randi(255,3,numel(cats));
for t=1:numel(testIdx)
  batch = testIdx(t);  
  
  scoret = [];
  for s=1:numel(opts.imageScales)
    for f=1:2 % add flips
      inputs = getBatch(bopts, imdb, batch, opts.imageScales(s), f-1 );
      net.eval(inputs) ;
  
      if isempty(scoret)
        scoret = squeeze(gather(net.vars(detLayer).value));
      else
        scoret = scoret + squeeze(gather(net.vars(detLayer).value));
      end
    end
  end
  
  % divide by number of scales and flips
  scoret = scoret / (2 * numel(opts.imageScales));
  im = imread(fullfile(imdb.imageDir,imdb.images.name{testIdx(t)}));
  
  for cls = 1:numel(cats)
    scores = scoret;
    boxes  = double(imdb.images.boxes{testIdx(t)});
    boxesSc = [boxes,scores(cls,:)'];
    boxesSc = boxesSc(boxesSc(:,5)>scTh,:);
    if isempty(boxesSc), continue; end;
    
    pick = nms(boxesSc, ovTh);
    boxesSc = boxesSc(pick,:);
    im = bbox_draw(im,boxesSc(1,1:4),rcolors(:,cls),2);
    fprintf('%s %.2f\n',cats{cls},boxesSc(1,5));
  end
  imshow(im);
  if exist('zs_dispFig', 'file'), zs_dispFig ; end
end



% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch, scale, flip)
% --------------------------------------------------------------------

opts.scale = scale;
opts.flip = flip;

images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
opts.prefetch = (nargout == 0);

[im,rois] = cnn_wsddn_get_batch(images, imdb, batch, opts);


rois = single(rois');
if opts.useGpu > 0
  im = gpuArray(im) ;
  rois = gpuArray(rois) ;
end

inputs = {'input', im, 'rois', rois} ;
  
  
if isfield(imdb.images,'boxScores')
  boxScore = reshape(imdb.images.boxScores{batch},[1 1 1 numel(imdb.images.boxScores{batch})]);
  inputs{end+1} = 'boxScore';
  inputs{end+1} = boxScore ; 
end

% -------------------------------------------------------------------------
function imdb = fixBBoxes(imdb, minSize, maxNum)

for i=1:numel(imdb.images.name)
  bbox = imdb.images.boxes{i};
  % remove small bbox
  isGood = (bbox(:,3)>=bbox(:,1)+minSize) & (bbox(:,4)>=bbox(:,2)+minSize);
  bbox = bbox(isGood,:);
  % remove duplicate ones
  [dummy, uniqueIdx] = unique(bbox, 'rows', 'first');
  uniqueIdx = sort(uniqueIdx);
  bbox = bbox(uniqueIdx,:);
  % limit number for training
  if imdb.images.set(i)~=3
    nB = min(size(bbox,1),maxNum);
  else
    nB = size(bbox,1);
  end
  
  if isfield(imdb.images,'boxScores')
    imdb.images.boxScores{i} = imdb.images.boxScores{i}(uniqueIdx);
    imdb.images.boxScores{i} = imdb.images.boxScores{i}(1:nB);
  end
  imdb.images.boxes{i} = bbox(1:nB,:);
  %   [h,w,~] = size(imdb.images.data{i});
  %   imdb.images.boxes{i} = [1 1 h w];
  
end

% -------------------------------------------------------------------------
function im = bbox_draw(im,roi,color,t)
% DRAWRECT
% IM : input image
% ROI : rectangle
% COLOR :
% T : thickness

[h,w,d] = size(im);
assert(d == numel(color));
if any(roi(:,1)>h) || any(roi(:,3)>h) || any(roi(:,2)>w) || any(roi(:,4)>w)
  error('Wrong bounding box coord!\n');
end
for c=1:d
  im(max(roi(1)-t,1):min(roi(1)+t,h),max(roi(2)-t,1):min(roi(4)+t,w),c) = color(c);
  im(max(roi(3)-t,1):min(roi(3)+t,h),max(roi(2)-t,1):min(roi(4)+t,w),c) = color(c);
  im(max(roi(1)-t,1):min(roi(3)+t,h),max(roi(2)-t,1):min(roi(2)+t,w),c) = color(c);
  im(max(roi(1)-t,1):min(roi(3)+t,h),max(roi(4)-t,1):min(roi(4)+t,w),c) = color(c);
end
