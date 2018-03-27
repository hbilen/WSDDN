function imdb = setup_voc07_ssw(varargin)
% setup_voc07_ssw  Initialize PASCAL VOC2007 data with selective
% search windows 

% Warning! boxes are in the format of ([y1 x1 y2 x2])

opts.dataDir = fullfile('data') ;
opts.proposalDir = fullfile(opts.dataDir,'SSW');
opts.loadTest = 1;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                 Load selective search win
% -------------------------------------------------------------------------
%% get selective search windows
files = {'SelectiveSearchVOC2007trainval.mat', ...
  'SelectiveSearchVOC2007test.mat'} ;

if ~exist(opts.proposalDir, 'dir')
  mkdir(opts.proposalDir) ;
end

for i=1:numel(files)
  if ~exist(fullfile(opts.proposalDir, files{i}), 'file')
    url = sprintf('http://koen.me/research/downloads/%s',files{i}) ;
    fprintf('downloading %s\n', url) ;
    urlwrite(url,[opts.proposalDir filesep files{i}]);
  end
end

if ~isempty(opts.proposalDir)
  t1 = load([opts.proposalDir,filesep,files{1}]);
  if opts.loadTest
    t2 = load([opts.proposalDir,filesep,files{2}]);
    ssw.id = [str2double(t1.images);str2double(t2.images)]';
    ssw.boxes = cat(2,t1.boxes,t2.boxes);
  else
    ssw.id = str2double(t1.images)';
    ssw.boxes = t1.boxes;
  end

  [~,si] = sort(ssw.id);
  ssw.id = ssw.id(si);
  ssw.boxes = ssw.boxes(si);
end

% -------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------
cats = {'aeroplane','bicycle','bird','boat','bottle','bus','car',...
  'cat','chair','cow','diningtable','dog','horse','motorbike','person',...
  'pottedplant','sheep','sofa','train','tvmonitor'};
    
if ~exist(opts.dataDir,'dir')
  error('wrong data folder!');
end

traindata = importdata(fullfile(opts.dataDir,'VOCdevkit','VOC2007','ImageSets','Main','train.txt'));
valdata = importdata(fullfile(opts.dataDir,'VOCdevkit','VOC2007','ImageSets','Main','val.txt'));
testdata = importdata(fullfile(opts.dataDir,'VOCdevkit','VOC2007','ImageSets','Main','test.txt'));

assert(numel(traindata)==2501);
assert(numel(valdata)==2510);
assert(numel(testdata)==4952);

imdb.classes.name = cats ;
imdb.classes.description = cats ;
imdb.imageDir = fullfile(opts.dataDir, fullfile('VOCdevkit','VOC2007','JPEGImages')) ;

% -------------------------------------------------------------------------
%                                                           Training images
% -------------------------------------------------------------------------% 
names = cell(1,numel(traindata));
labels = zeros(numel(traindata),numel(cats));


% load image names
for t=1:numel(traindata)
  names{t} = sprintf('%06d.jpg',traindata(t));
%   data{t} = imread(sprintf('%s/%s',imdb.imageDir,names{t}));
end

% load binary labels
for c=1:numel(cats)
  t = importdata(fullfile(opts.dataDir,'VOCdevkit','VOC2007','ImageSets','Main',[cats{c},'_train.txt']));
  labels(:,c) = t(:,2);
end

imdb.images.id = traindata';
imdb.images.name = names ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels' ;
% imdb.images.data = data;

% -------------------------------------------------------------------------
%                                                         Validation images
% -------------------------------------------------------------------------

names = cell(1,numel(valdata));
labels = zeros(numel(valdata),numel(cats));
% data = cell(1,numel(valdata));

% load image names
for t=1:numel(valdata)
  names{t} = sprintf('%06d.jpg',valdata(t));
%   data{t} = imread(sprintf('%s/%s',imdb.imageDir,names{t}));
end

% load binary labels
for c=1:numel(cats)
  t = importdata(fullfile(opts.dataDir,'VOCdevkit','VOC2007','ImageSets','Main',[cats{c},'_val.txt']));
  labels(:,c) = t(:,2);
end


imdb.images.id = horzcat(imdb.images.id, valdata') ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels') ;
% imdb.images.data = horzcat(imdb.images.data, data) ;

% % -------------------------------------------------------------------------
% %                                                               Test images
% % -------------------------------------------------------------------------
% 
%
if opts.loadTest
  names = cell(1,numel(testdata));
  labels = zeros(numel(testdata),numel(cats));
  % data = cell(1,numel(testdata));

  % load image names
  for t=1:numel(testdata)
    names{t} = sprintf('%06d.jpg',testdata(t));
  %   data{t} = imread(sprintf('%s/%s',imdb.imageDir,names{t}));
  end

  % load binary labels
  for c=1:numel(cats)
    t = importdata(fullfile(opts.dataDir,'VOCdevkit','VOC2007','ImageSets','Main',[cats{c},'_test.txt']));
    labels(:,c) = t(:,2);
  end

  imdb.images.id = horzcat(imdb.images.id, testdata') ;
  imdb.images.name = horzcat(imdb.images.name, names) ;
  imdb.images.set = horzcat(imdb.images.set, 3 * ones(1,numel(names))) ;
  imdb.images.label = horzcat(imdb.images.label, labels') ;
  % imdb.images.data = horzcat(imdb.images.data, data) ;
end
% -------------------------------------------------------------------------
%                                                            Postprocessing
% -------------------------------------------------------------------------
[~,sorti] = sort(imdb.images.id);


imdb.images.id = imdb.images.id(sorti);
imdb.images.name = imdb.images.name(sorti) ;
imdb.images.set = imdb.images.set(sorti) ;
imdb.images.label = single(imdb.images.label(:,sorti)) ;
imdb.images.size = zeros(numel(imdb.images.name),2);

if ~isempty(opts.proposalDir)
  imdb.images.boxes = ssw.boxes;
  assert(all(ssw.id==imdb.images.id));
end

% this is zero as scores of selective search windows are not much
% informative
if ~isempty(opts.proposalDir)
imdb.images.boxScores = cell(size(imdb.images.boxes));
for i=1:numel(imdb.images.boxes)
  imdb.images.boxes{i} = int16(imdb.images.boxes{i});
  imdb.images.boxScores{i} = zeros(size(imdb.images.boxes{i},1),1,'single');
  imf = imfinfo(fullfile(imdb.imageDir,imdb.images.name{i}));
  imdb.images.size(i,:) = [imf.Height,imf.Width];
end
end
end
