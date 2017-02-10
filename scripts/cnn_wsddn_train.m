function [net, info] = cnn_wsddn_train(varargin)
% @author: Hakan Bilen
% cnn_wsddn_train: training script for WSDDN

addpath('layers');
addpath(fullfile('layers','matlab'));

mt = 'matconvnet';
run(fullfile(mt, 'matlab', 'vl_setupnn.m')) ;
af = strcat(mt,filesep,{'examples'});
for a=1:numel(af)
  addpath(af{a});
end

opts.dataDir = fullfile('data') ;
opts.expDir = fullfile('exp') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.modelPath = fullfile('models', 'imagenet-vgg-f.mat') ;
% opts.proposalDir = fullfile('data','SSW');
opts.proposalDir = fullfile('data','EB');

opts.addBiasSamples = 1; % add Box Scores
opts.addLossSmooth  = 1; % add Spatial Regulariser
opts.softmaxTempCls = 1; % softmax temp for cls
opts.softmaxTempDet = 2; % softmax temp for det
% if you have limited gpu memory (<6gb), you can change the next 2 params
opts.maxNumProposals = inf; % limit number (eg 1500)
opts.imageScales = [480,576,688,864,1200]; % scales
% 
opts.minBoxSize = 20; % minimum bounding box size


opts.train.gpus = [] ;
opts.train.batchSize = 1 ;
opts.train.numSubBatches = 1;
opts.train.continue = true ;
opts.train.prefetch = true;
opts.train.learningRate = 1e-5 * [ones(1,10) 0.1*ones(1,10)] ;
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = 20;
opts.train.derOutputs = {'objective', 1} ;

opts.numFetchThreads = 2 ;
opts = vl_argparse(opts, varargin) ;


display(opts);

opts.train.expDir = opts.expDir ;
opts.train.numEpochs = numel(opts.train.learningRate) ;
%% ------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
nopts.addBiasSamples = opts.addBiasSamples; % add Box Scores (only with Edge Boxes)
nopts.addLossSmooth  = opts.addLossSmooth; % add Spatial Regulariser
nopts.softmaxTempCls = opts.softmaxTempCls; % softmax temp for cls
nopts.softmaxTempDet = opts.softmaxTempDet; % softmax temp for det

net = load(opts.modelPath);
net = prepare_wsddn(net,nopts);
opts.train.derOutputs = {'objective', 1};
if nopts.addLossSmooth
  opts.train.derOutputs{end+1} = 'lossTopB';
  opts.train.derOutputs{end+1} = 1e-4 ;
end

if isfield(net,'normalization')
  net.normalization.averageImage = ...
    reshape(single([102.9801, 115.9465, 122.7717]),[1 1 3]);
  bopts = net.normalization;
else
  net.meta.normalization.averageImage = ...
    reshape(single([102.9801, 115.9465, 122.7717]),[1 1 3]);
  bopts = net.meta.normalization;
end
bopts.interpolation = 'bilinear';
bopts.imageScales = opts.imageScales;
bopts.numThreads = opts.numFetchThreads;
%% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------
fprintf('loading imdb...');
if exist(opts.imdbPath,'file')==2
  imdb = load(opts.imdbPath) ;
else
%   imdb = cnn_voc07_ssw_setup_data('dataDir',opts.dataDir, ...
%     'proposalDir',opts.proposalDir,'loadTest',1);
  imdb = cnn_voc07_eb_setup_data('dataDir',opts.dataDir, ...
    'proposalDir',opts.proposalDir,'loadTest',1);
  
  imdbFolder = fileparts(opts.imdbPath);
  
  if ~exist(imdbFolder,'dir')
    mkdir(imdbFolder);
  end
  save(opts.imdbPath,'-struct', 'imdb', '-v7.3');
end

fprintf('done\n');

imdb = fixBBoxes(imdb, opts.minBoxSize, opts.maxNumProposals);

%% -------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
% use train + val for training
imdb.images.set(imdb.images.set == 2) = 1;
trainIdx = find(imdb.images.set == 1);

% avoid test data
% valIdx = find(imdb.images.set == 3);
valIdx = [];

%% 
bopts.useGpu = numel(opts.train.gpus) >  0 ;
bopts.prefetch = opts.train.prefetch;
bopts.addLossSmooth = opts.addLossSmooth;

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
  return;
end

opts.scale = opts.imageScales(randi(numel(opts.imageScales)));
opts.flip = randi(2,numel(batch),1)-1; % random flip

images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
opts.prefetch = (nargout == 0);

[im,rois] = cnn_wsddn_get_batch(images, imdb, batch, opts);

labels = imdb.images.label(:,batch) ;
labels = reshape(labels,[1 1 size(labels,1) numel(batch)]);

rois = single(rois');
if opts.useGpu > 0
  im = gpuArray(im) ;
  rois = gpuArray(rois) ;
end

inputs = {'input', im, 'label', labels, 'rois', rois, 'ids', batch};

if bopts.addLossSmooth
  inputs{end+1} = 'boxes';
  inputs{end+1} = imdb.images.boxes{batch} ;
end
  
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
    imdb.images.boxScores{i} = imdb.images.boxScores{i}(isGood);
    imdb.images.boxScores{i} = imdb.images.boxScores{i}(uniqueIdx);
    imdb.images.boxScores{i} = imdb.images.boxScores{i}(1:nB);
  end
  imdb.images.boxes{i} = bbox(1:nB,:);
  %   [h,w,~] = size(imdb.images.data{i});
  %   imdb.images.boxes{i} = [1 1 h w];
  
end
