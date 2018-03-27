function [net, info] = wsddn_train(varargin)
% @author: Hakan Bilen
% wsddn_train: training script for WSDDN

run(fullfile('matconvnet', 'matlab', 'vl_setupnn.m')) ;
addpath('matconvnet','examples') ;
addpath('matconvnet','examples','imagenet') ;

opts.dataDir = fullfile('data') ;
opts.expDir = fullfile('exp') ;
opts.imdbPath = fullfile('imdbs', 'imdb-eb.mat');
opts.modelPath = fullfile('models', 'imagenet-vgg-f.mat') ;
opts.proposalType = 'eb' ;
opts.proposalDir = fullfile('data','EdgeBoxes') ;


opts.addBiasSamples = 1; % add Box Scores
opts.addLossSmooth  = 1; % add Spatial Regulariser
opts.softmaxTempCls = 1; % softmax temp for cls
opts.softmaxTempDet = 2; % softmax temp for det
opts.maxScale = 2000 ;

% if you have limited gpu memory (<6gb), you can change the next 2 params
opts.maxNumProposals = inf; % limit number (eg 1500)
opts.imageScales = [480,576,688,864,1200]; % scales
opts.minBoxSize = 20; % minimum bounding box size
opts.train.gpus = [] ;
opts.train.continue = true ;
opts.train.prefetch = true ;
opts.train.learningRate = 1e-5 * [ones(1,10) 0.1*ones(1,10)] ;
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = 20;
opts.train.derOutputs = {'objective', 1} ;

opts.numFetchThreads = 1 ;
opts = vl_argparse(opts, varargin) ;

display(opts);

opts.train.batchSize = 1 ;
opts.train.expDir = opts.expDir ;
opts.train.numEpochs = numel(opts.train.learningRate) ;
%% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------
fprintf('loading imdb...');
if exist(opts.imdbPath,'file')==2
  imdb = load(opts.imdbPath) ;
else
  if strcmp(opts.proposalType,'ssw')
    imdb = cnn_voc07_ssw_setup_data('dataDir',opts.dataDir, ...
      'proposalDir',opts.proposalDir,'loadTest',1);
  elseif strcmp(opts.proposalType,'eb')
  imdb = cnn_voc07_eb_setup_data('dataDir',opts.dataDir, ...
    'proposalDir',opts.proposalDir,'loadTest',1);
  else
    error('undefined proposal type %s\n',opts.proposalType)
  end
  
  imdbFolder = fileparts(opts.imdbPath);
  
  if ~exist(imdbFolder,'dir')
    mkdir(imdbFolder);
  end
  save(opts.imdbPath,'-struct', 'imdb', '-v7.3');
end

fprintf('done\n');

imdb = fixBBoxes(imdb, opts.minBoxSize, opts.maxNumProposals);

% use train + val for training
imdb.images.set(imdb.images.set == 2) = 1;
trainIdx = find(imdb.images.set == 1);

%% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.dataDir, 'imageStats.mat') ;
if exist(imageStatsPath,'file')
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
 
  images = imdb.images.name(imdb.images.set == 1) ;
  images = strcat([imdb.imageDir filesep],images) ;
  
  [averageImage, rgbMean, rgbCovariance] = getImageStats(images, ...
    'imageSize', [256 256], ...
    'numThreads', opts.numFetchThreads, ...
    'gpus', opts.train.gpus) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end
[v,d] = eig(rgbCovariance) ;
rgbDeviation = v*sqrt(d) ;
clear v d ;


%% ------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
nopts.addBiasSamples = opts.addBiasSamples; % add Box Scores (only with Edge Boxes)
nopts.addLossSmooth  = opts.addLossSmooth; % add Spatial Regulariser
nopts.softmaxTempCls = opts.softmaxTempCls; % softmax temp for cls
nopts.softmaxTempDet = opts.softmaxTempDet; % softmax temp for det

nopts.averageImage = reshape(rgbMean,[1 1 3]) ;
nopts.rgbVariance = 0.1 * rgbDeviation ;
nopts.numClasses = numel(imdb.classes.name) ;
nopts.classNames = imdb.classes.name ;

net = load(opts.modelPath);
net = wsddn_init(net,nopts);

if nopts.addLossSmooth
  opts.train.derOutputs = {'objective', 1, 'lossTopB', 1e-4} ;
end


if ~exist(opts.expDir,'dir')
  mkdir(opts.expDir) ;
end

%% -------------------------------------------------------------------------
%                                                   Database stats
% -------------------------------------------------------------------------
bopts = net.meta.normalization;
net.meta.augmentation.jitterBrightness = 0 ;
% bopts.interpolation = 'bilinear';
bopts.jitterBrightness = net.meta.augmentation.jitterBrightness ;
bopts.imageScales = opts.imageScales;
bopts.numThreads = opts.numFetchThreads;
bopts.addLossSmooth = opts.addLossSmooth;
bopts.addBiasSamples = opts.addBiasSamples;
bopts.maxScale = opts.maxScale ;
bopts.vgg16 = any(arrayfun(@(a) strcmp(a.name, 'relu5'), net.layers)==1) ;
%% -------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
% avoid test data
valIdx = find(imdb.images.set == 3);
valIdx = valIdx(1:5:end) ;
% valIdx = [];

%% 
bopts.useGpu = numel(opts.train.gpus) >  0 ;
bopts.prefetch = opts.train.prefetch;

info = cnn_train_dag(net, imdb, @(i,b) ...
  getBatch(bopts,i,b), ...
  opts.train, 'train', trainIdx, ...
  'val', valIdx) ;

%% -------------------------------------------------------------------
%                                                       Deploy network
% --------------------------------------------------------------------
removeLoss = {'LossTopBoxSmooth','loss','mAP'};
for i=1:numel(removeLoss)
  if sum(arrayfun(@(a) strcmp(a.name, removeLoss{i}), net.layers)==1)
    net.removeLayer(removeLoss{i});
  end
end

net.mode = 'test' ;
net_ = net ;
net = net_.saveobj() ;
save(fullfile(opts.expDir,'net.mat'), '-struct','net');

% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------
if isempty(batch)
  inputs = {'input', [], 'label', [], 'rois', [], 'ids', []};
  return;
end

opts.scale = opts.imageScales(randi(numel(opts.imageScales)));
opts.flip = randi(2,numel(batch),1)-1; % random flip
is_vgg16 = opts.vgg16 ;
opts = rmfield(opts,'vgg16') ;

images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
opts.prefetch = (nargout == 0);

[im,rois] = wsddn_get_batch(images, imdb, batch, opts);

if nargout>0
  rois = single(rois') ;
  labels = imdb.images.label(:,batch) ;
  labels = reshape(labels,[1 1 size(labels,1) numel(batch)]);

  if opts.useGpu > 0
    im = gpuArray(im) ;
    rois = gpuArray(rois) ;
  end

  if ~isempty(rois)
   rois = rois([1 3 2 5 4],:) ;
  end

  ss = [16 16] ;

  if is_vgg16
    o0 = 8.5 ;
    o1 = 9.5 ;
  else
    o0 = 18 ;
    o1 = 9.5 ;
  end

  rois = [ rois(1,:); ...
    floor((rois(2,:) - o0 + o1) / ss(1) + 0.5) + 1;
    floor((rois(3,:) - o0 + o1) / ss(2) + 0.5) + 1;
    ceil((rois(4,:) - o0 - o1) / ss(1) - 0.5) + 1;
    ceil((rois(5,:) - o0 - o1) / ss(2) - 0.5) + 1];


  inputs = {'input', im, 'label', labels, 'rois', rois, 'ids', batch} ;

  if opts.addLossSmooth
    inputs{end+1} = 'boxes' ;
    inputs{end+1} = imdb.images.boxes{batch} ;
  end

  if opts.addBiasSamples==1
    boxScore = reshape(imdb.images.boxScores{batch},[1 1 1 numel(imdb.images.boxScores{batch})]);
    inputs{end+1} = 'boxScore';
    inputs{end+1} = boxScore ;
  end
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
    imdb.images.boxScores{i} = imdb.images.boxScores{i}(isGood);
    imdb.images.boxScores{i} = imdb.images.boxScores{i}(uniqueIdx);
    imdb.images.boxScores{i} = imdb.images.boxScores{i}(1:nB);
  end
  imdb.images.boxes{i} = bbox(1:nB,:);
  %   [h,w,~] = size(imdb.images.data{i});
  %   imdb.images.boxes{i} = [1 1 h w];
  
end
