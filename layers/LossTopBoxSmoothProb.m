classdef LossTopBoxSmoothProb < dagnn.Loss
  % given top scoring box, it finds other boxes with at least overlap of
  % minOverlap and calculates the euclidean dist between top and other
  % boxes
  
  properties (Transient)
    gtIdx = []
    boxIdx = []
    probs = []
    minOverlap = 0.5
    nBoxes = 10
  end
  
  methods
    function outputs = forward(obj, inputs, params)
      if numel(inputs) ~= 4
        error('Number of inputs is not 2');
      end
      obj.gtIdx = [];
      obj.boxIdx = [];
      obj.probs = [];
      boxes  = gather(inputs{2})';
      scores = gather(squeeze(inputs{3}));
      labels = gather(squeeze(inputs{4}));
      
      if numel(boxes)<5
        return;
      end
      
      outputs{1} = zeros(1,'like',inputs{1});
      for c=1:numel(labels)
        if labels(c)<=0
          continue;
        end
        
        [so, si] = sort(scores(c,:),'descend');
        obj.gtIdx{c} = si(1);
        gtBox = boxes(:,obj.gtIdx{c});
        gtArea = (gtBox(3)-gtBox(1)+1) .* (gtBox(4)-gtBox(2)+1);
        
        bbs = boxes(:,si(2:min(obj.nBoxes,end)))';
        
        y1 = bbs(:,1);
        x1 = bbs(:,2);
        y2 = bbs(:,3);
        x2 = bbs(:,4);
        
        area = (x2-x1+1) .* (y2-y1+1);
        
        yy1 = max(gtBox(1), y1);
        xx1 = max(gtBox(2), x1);
        yy2 = min(gtBox(3), y2);
        xx2 = min(gtBox(4), x2);
        
        w = max(0.0, xx2-xx1+1);
        h = max(0.0, yy2-yy1+1);
        
        inter = w.*h;
        o = find((inter ./ (gtArea + area - inter))>obj.minOverlap);
        
        if isempty(o)
          continue;
        end
        
        obj.boxIdx{c} = si(o+1);
        obj.probs{c} = so(o+1);
        d = bsxfun(@minus,inputs{1}(:,:,:,obj.boxIdx{c}),inputs{1}(:,:,:,obj.gtIdx{c}));
        d = bsxfun(@times,d,obj.probs{c});
        outputs{1} = outputs{1} + 0.5 * sum(d(:).^2);
      end
      
      n = obj.numAveraged ;
      m = n + 1 ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1,4) ;
      derInputs{1} = zeros(size(inputs{1}),'like',inputs{1});
      for c=1:numel(obj.boxIdx)
        if isempty(obj.boxIdx{c}), continue; end
        derInputs{1}(:,:,:,obj.boxIdx{c}) = ...
          bsxfun(@minus,inputs{1}(:,:,:,obj.boxIdx{c}),inputs{1}(:,:,:,obj.gtIdx{c}));
        derInputs{1}(:,:,:,obj.boxIdx{c}) = bsxfun(@times,...
          reshape(obj.probs{c},[1 1 1 numel(obj.probs{c})]),derInputs{1}(:,:,:,obj.boxIdx{c}));
        derInputs{1}(:,:,:,obj.gtIdx{c}) = -sum(derInputs{1}(:,:,:,obj.boxIdx{c}),4);

      end
      derInputs{1} = derInputs{1} * derOutputs{1};
%       fprintf('LossTopBox l2 %f ',sqrt(sum(derInputs{1}(:).^2)));
      derParams = {} ;
    end
    
    function obj = LossTopBoxSmoothProb(varargin)
      obj.load(varargin) ;
    end
    
    function reset(obj)
      obj.gtIdx = [];
      obj.boxIdx = [];
      obj.probs = [];
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end
    
    
  end
  
end
