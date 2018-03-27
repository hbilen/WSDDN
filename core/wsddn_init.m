% --------------------------------------------------------------------
function net = wsddn_init(net,varargin)
% --------------------------------------------------------------------
% @author: Hakan Bilen
% wsddn_init : this script initalise WSDDN model

opts.addBiasSamples = 1 ;
opts.softmaxTempCls = 1 ;
opts.softmaxTempDet = 2 ;
opts.addLossSmooth  = 1 ;
opts.averageImage = [] ;
opts.rgbVariance = [] ;
opts.numClasses = 1 ;
opts.classNames = {''} ;

opts = vl_argparse(opts, varargin) ;

% add drop-out layers
relu6p = find(cellfun(@(a) strcmp(a.name, 'relu6'), net.layers)==1);
relu7p = find(cellfun(@(a) strcmp(a.name, 'relu7'), net.layers)==1);

drop6 = struct('type', 'dropout', 'rate', 0.5, 'name','drop6');
drop7 = struct('type', 'dropout', 'rate', 0.5, 'name','drop7');
net.layers = [net.layers(1:relu6p) drop6 net.layers(relu6p+1:relu7p) drop7 net.layers(relu7p+1:end)];


% change loss fc layer
fc8p = (cellfun(@(a) strcmp(a.name, 'fc8'), net.layers)==1);
net.layers{fc8p}.weights{1} = 0.01 * ...
  randn(1,1,size(net.layers{fc8p}.weights{1},3),opts.numClasses,'single');

net.layers{fc8p}.weights{2} = zeros(1, opts.numClasses, 'single');
net.layers{fc8p}.name = 'fc8C';

net.layers(end) = [] ;
% add loss (this will be changed to binary log at the end)
% net.layers{end} = struct('name','loss', 'type','softmaxloss') ;

% add detection layer
clsLayerPos  = (cellfun(@(a) strcmp(a.name, 'fc8C'), net.layers)==1);
detLayer = net.layers{clsLayerPos};
detLayer.weights{1} = 0.01 * randn(1,1,size(detLayer.weights{1},3),opts.numClasses,'single');
% detLayer.weights{1} = zeros(1,1,size(detLayer.weights{1},3),opts.numClasses,'single');
detLayer.weights{2} = zeros(1, opts.numClasses, 'single');

detLayer.name = 'fc8R';

% remove pool5
pPool5 = find(cellfun(@(a) strcmp(a.name, 'pool5'), net.layers)==1);
net.layers = [net.layers([1:pPool5-1,pPool5+1:end]) detLayer];

% convert to dagnn
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

% fix fc8R
pFc8R = (arrayfun(@(a) strcmp(a.name, 'fc8R'), net.layers)==1);
pFc8C = (arrayfun(@(a) strcmp(a.name, 'fc8C'), net.layers)==1);

net.layers(pFc8R).inputs = net.layers(pFc8C).inputs;
net.layers(pFc8R).inputIndexes = net.layers(pFc8C).inputIndexes;

% add spp

pRelu5 = (arrayfun(@(a) strcmp(a.name, 'relu5'), net.layers)==1);
vggdeep = 0;
if all(pRelu5==0)
  pRelu5 = (arrayfun(@(a) strcmp(a.name, 'relu5_3'), net.layers)==1);
  assert(any(pRelu5==1));
  vggdeep = 1;
end
pFc6 = (arrayfun(@(a) strcmp(a.name, 'fc6'), net.layers)==1);

% add spp (offset1 = rf offset, offset2 = shrinking factor)
% offset1=18  offset2=9.5 levels=6 for vgg-f and vgg-m-1024
% offset1=8.5 offset2=9.5 levels=7 for vgg-very-deep-16
if vggdeep
  net.addLayer('SPP', dagnn.ROIPooling('subdivisions',[7 7],...
    'transform',1), ...
    {net.layers(pRelu5).outputs{1},'rois'}, ...
    'xSPP');
else
  net.addLayer('SPP', dagnn.ROIPooling('subdivisions',[6 6],...
    'transform',1), ...
    {net.layers(pRelu5).outputs{1},'rois'}, ...
    'xSPP');
end


if opts.addBiasSamples
  % add boost
  net.addLayer('boostBox', ...
    dagnn.BiasSamples('scale',10), ...
    {'xSPP','boxScore'},'xBoostBox');
  net.layers(pFc6).inputs{1} = 'xBoostBox';
else
  net.layers(pFc6).inputs{1} = 'xSPP';
end



% add softmax layer for det
pFc8R = (arrayfun(@(a) strcmp(a.name, 'fc8R'), net.layers)==1);
net.addLayer('softmaxDet', ...
  dagnn.SoftMax2('dim',4, 'temp',opts.softmaxTempDet), ...
  net.layers(pFc8R).outputs{1},'xSoftmaxDet');

% add softmax layers for cls
pFc8C = (arrayfun(@(a) strcmp(a.name, 'fc8C'), net.layers)==1);
net.layers(pFc8C).outputs{1} = 'xfc8C';

net.addLayer('softmaxCls', ...
  dagnn.SoftMax2('dim',3, 'temp',opts.softmaxTempCls), ...
  net.layers(pFc8C).outputs{1},'xSoftmaxCls');

% add times layer
net.addLayer('timesCR', ...
  dagnn.Times(), ...
  {'xSoftmaxCls','xSoftmaxDet'},'xTimes');

% add sum layer
net.addLayer('sum', ...
  dagnn.SumOverDim('dim',4), ...
  'xTimes','prediction');



% add classification AP
net.addLayer('mAP', dagnn.LayerAP('cls_index',1:opts.numClasses), ...
  {'prediction','label', 'ids'}, 'mAP') ;

net.addLayer('loss', dagnn.Loss('loss','binarylog'), ...
  {'prediction','label'}, 'objective') ;


% no decay for bias
for i=2:2:numel(net.params)
  net.params(i).weightDecay = 0;
end

if opts.addLossSmooth
  net.addLayer('LossTopBoxSmooth',dagnn.LossTopBoxSmoothProb('minOverlap',0.6),...
    {net.layers(pFc8R).inputs{1},'boxes','xTimes','label'},...
    'lossTopB');
end
meta = net.meta ; 
net.meta = [] ;
net.meta.normalization.interpolation = meta.normalization.interpolation ;
net.meta.normalization.averageImage  = opts.averageImage ;
net.meta.normalization.rgbVariance   = opts.rgbVariance ;