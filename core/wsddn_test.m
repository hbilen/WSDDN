function aps = wsddn_test(varargin)
% @author: Hakan Bilen
% wsddn_test : this script evaluates detection performance in PASCAL VOC
% dataset for given a WSDDN model

opts.dataDir = fullfile(vl_rootnn, 'data') ;
opts.expDir = fullfile(vl_rootnn, 'exp') ;
opts.imdbPath = fullfile(vl_rootnn, 'data', 'imdbs', 'imdb-eb.mat');
opts.modelPath = fullfile(vl_rootnn, 'exp', 'net.mat') ;
opts.proposalType = 'eb' ;
opts.proposalDir = fullfile(vl_rootnn, 'data','EdgeBoxes') ;

% if you have limited gpu memory (<6gb), you can change the next 2 params
opts.maxNumProposals = inf; % limit number
opts.imageScales = [480,576,688,864,1200]; % scales

opts.gpu = [] ;
opts.train.prefetch = true ;
opts.vis = 0 ;
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
% figure(2) ;
if isfield(net,'net')
  net = net.net;
end
net = dagnn.DagNN.loadobj(net) ;

net.mode = 'test' ;
if ~isempty(opts.gpu)
  gpuDevice(opts.gpu) ;
  net.move('gpu') ;
end

if isfield(net,'normalization')
  bopts = net.normalization;
else
  bopts = net.meta.normalization;
end

bopts.rgbVariance = [] ;
bopts.interpolation = net.meta.normalization.interpolation;
bopts.jitterBrightness = 0 ;
bopts.imageScales = opts.imageScales;
bopts.numThreads = opts.numFetchThreads;
bs = find(arrayfun(@(a) isa(a.block, 'dagnn.BiasSamples'), net.layers)==1);
bopts.addBiasSamples = ~isempty(bs) ;
bopts.vgg16 = any(arrayfun(@(a) strcmp(a.name, 'relu5'), net.layers)==1) ;
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

VOCinit;
VOCopts.testset = 'test';
VOCopts.annopath = fullfile(opts.dataDir,'VOCdevkit','VOC2007','Annotations','%s.xml');
VOCopts.imgsetpath = fullfile(opts.dataDir,'VOCdevkit','VOC2007','ImageSets','Main','%s.txt');
VOCopts.localdir = fullfile(opts.dataDir,'VOCdevkit','local','VOC2007');
cats = VOCopts.classes;
ovTh = 0.4;
scTh = 1e-3;
% --------------------------------------------------------------------
%                                                               Detect
% --------------------------------------------------------------------
if strcmp(VOCopts.testset,'test')
  testIdx = find(imdb.images.set == 3);
elseif strcmp(VOCopts.testset,'trainval')
  testIdx = find(imdb.images.set < 3);
end
bopts.useGpu = numel(opts.gpu) >  0 ;

scores = cell(1,numel(testIdx));
boxes = imdb.images.boxes(testIdx);
names = imdb.images.name(testIdx);

detLayer = find(arrayfun(@(a) strcmp(a.name, 'xTimes'), net.vars)==1);
net.vars(detLayer(1)).precious = 1;
% run detection
start = tic ;
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
  scores{t} = scoret;
  % show speed
  time = toc(start) ;
  n = t * 2 * numel(opts.imageScales) ; % number of images processed overall
  speed = n/time ;
  if mod(t,10)==0
    fprintf('test %d / %d speed %.1f Hz\n',t,numel(testIdx),speed);
  end
  
  
  if opts.vis
    for cls = 1:numel(cats)
      idx = (scores{t}(cls,:)>0.05);
      if sum(idx)==0, continue;end
        % divide by number of scales and flips
  
      im = imread(fullfile(imdb.imageDir,imdb.images.name{testIdx(t)}));
      boxest  = double(imdb.images.boxes{testIdx(t)}(idx,:));
      scorest = scores{t}(cls,idx)' / (2 * numel(opts.imageScales));
      boxesSc = [boxest,scorest];
      pick = nms(boxesSc, ovTh);
      boxesSc = boxesSc(pick,:);
      figure(1) ;
      im = bbox_draw(im,boxesSc(1,[2 1 4 3 5]));
      fprintf('%s %.2f',cats{cls},boxesSc(1,5));
     
      fprintf('\n') ;
      title(cats{cls});
      pause;

    end
  end  
end

dets.names  = names;
dets.scores = scores;
dets.boxes  = boxes;

% --------------------------------------------------------------------
%                                                PASCAL VOC evaluation
% --------------------------------------------------------------------

aps = zeros(numel(cats),1);
for cls = 1:numel(cats)
  
  vocDets.confidence = [];
  vocDets.bbox       = [];
  vocDets.ids        = [];

  for i=1:numel(dets.names)
    
    scores = double(dets.scores{i});
    boxes  = double(dets.boxes{i});
    
    boxesSc = [boxes,scores(cls,:)'];
    boxesSc = boxesSc(boxesSc(:,5)>scTh,:);
    pick = nms(boxesSc, ovTh);
    boxesSc = boxesSc(pick,:);
    
    vocDets.confidence = [vocDets.confidence;boxesSc(:,5)];
    vocDets.bbox = [vocDets.bbox;boxesSc(:,[2 1 4 3])];
    vocDets.ids = [vocDets.ids; repmat({dets.names{i}(1:6)},size(boxesSc,1),1)];
    
  end
  [rec,prec,ap] = wsddnVOCevaldet(VOCopts,cats{cls},vocDets,0);
  
  fprintf('%s %.1f\n',cats{cls},100*ap);
  aps(cls) = ap;
end

% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch, scale, flip)
% --------------------------------------------------------------------

opts.scale = scale;
opts.flip = flip;
is_vgg16 = opts.vgg16 ;
opts = rmfield(opts,'vgg16') ;

images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
opts.prefetch = (nargout == 0);

[im,rois] = wsddn_get_batch(images, imdb, batch, opts);


rois = single(rois');
if opts.useGpu > 0
  im = gpuArray(im) ;
  rois = gpuArray(rois) ;
end
rois = rois([1 3 2 5 4],:) ;


ss = [16 16] ;
if is_vgg16
  o0 = 8.5 ;
  o1 = 9.5 ;
else
  o0 = 18 ;
  o1 = 9.5 ;
end
rois = [ rois(1,:);
        floor((rois(2,:) - o0 + o1) / ss(1) + 0.5) + 1;
        floor((rois(3,:) - o0 + o1) / ss(2) + 0.5) + 1;
        ceil((rois(4,:) - o0 - o1) / ss(1) - 0.5) + 1;
        ceil((rois(5,:) - o0 - o1) / ss(2) - 0.5) + 1];

      
inputs = {'input', im, 'rois', rois} ;
  
  
if opts.addBiasSamples && isfield(imdb.images,'boxScores')
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

%-------------------------------------------------------------------------%

function im = bbox_draw(im,boxes,c,t)

% copied from Ross Girshick
% Fast R-CNN
% Copyright (c) 2015 Microsoft
% Licensed under The MIT License [see LICENSE for details]
% Written by Ross Girshick
% --------------------------------------------------------
% source: https://github.com/rbgirshick/fast-rcnn/blob/master/matlab/showboxes.m
%
%
% Fast R-CNN
% 
% Copyright (c) Microsoft Corporation
% 
% All rights reserved.
% 
% MIT License
% 
% Permission is hereby granted, free of charge, to any person obtaining a
% copy of this software and associated documentation files (the "Software"),
% to deal in the Software without restriction, including without limitation
% the rights to use, copy, modify, merge, publish, distribute, sublicense,
% and/or sell copies of the Software, and to permit persons to whom the
% Software is furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
% THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
% OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
% ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
% OTHER DEALINGS IN THE SOFTWARE.

image(im);
axis image;
axis off;
set(gcf, 'Color', 'white');

if nargin<3
  c = 'r';
  t = 2;
end

s = '-';
if ~isempty(boxes)
    x1 = boxes(:, 1);
    y1 = boxes(:, 2);
    x2 = boxes(:, 3);
    y2 = boxes(:, 4);
    line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', ...
        'color', c, 'linewidth', t, 'linestyle', s);
    for i = 1:size(boxes, 1)
        text(double(x1(i)), double(y1(i)) - 2, ...
            sprintf('%.4f', boxes(i, end)), ...
            'backgroundcolor', 'b', 'color', 'w', 'FontSize', 8);
    end
end
