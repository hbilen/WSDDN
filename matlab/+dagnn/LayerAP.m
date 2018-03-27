classdef LayerAP < dagnn.Loss
  % @author: Hakan Bilen
  % 11 step average precision
  properties
    cls_index = 1
    resetLayer = false 
    gtLabels = []
    scores   = []
    ids      = []
    aps      = []
    voc07    = true % 11 step
    classNames = {} 
  end


  methods
    function outputs = forward(obj, inputs, params)
      if obj.resetLayer 
        obj.gtLabels = [] ;
        obj.scores   = [] ;
        obj.ids      = [] ;
        obj.aps      = [] ;
        obj.resetLayer = false ;
      end
      
      if numel(inputs)==2
        obj.scores = [obj.scores gather(squeeze(inputs{1}(:,:,obj.cls_index,:)))];
        obj.gtLabels = [obj.gtLabels gather(squeeze(inputs{2}(:,:,obj.cls_index,:)))];
      elseif numel(inputs)>2
        scoresCur = gather(squeeze(inputs{1}(:,:,obj.cls_index,:)));
        gtLabelsCur = gather(squeeze(inputs{2}(:,:,obj.cls_index,:)));
        
        idsCur = gather(squeeze(inputs{3}));
        
        [lia,locb] = ismember(idsCur,obj.ids);
        
        if any(lia)
          obj.scores = [obj.scores scoresCur(~lia,:)];
          obj.gtLabels = [obj.gtLabels gtLabelsCur(~lia,:)];
          obj.ids = [obj.ids(:) ; idsCur(~lia,:)];
          
          nz = find(lia);
          for i=1:numel(nz)
            obj.scores(locb(nz(i)),:) = obj.scores(locb(nz(i)),:) + ...
              scoresCur(nz(i),:);
          end
        else
          obj.scores = [obj.scores scoresCur];
          obj.gtLabels = [obj.gtLabels gtLabelsCur];
          obj.ids = [obj.ids(:) ; idsCur]';
        end
      else
        error('wrong number of inputs');
      end
      
      obj.aps = obj.compute_average_precision();
      obj.average = 100 * mean(obj.aps);
      outputs{1} =  100 * mean(obj.aps);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1,numel(inputs));
      derInputs{1} = derOutputs{1} ;
      derParams = {} ;
    end

    function reset(obj)
      obj.resetLayer = true ;
%       obj.average = 0 ;
%       obj.aps = 0 ;
%       obj.gtLabels = [];
%       obj.scores   = [];
%       obj.ids      = [];
    end

    function printAP(obj)
      if isempty(obj.classNames)
        for i=1:numel(obj.aps)
          fprintf('class-%d %.1f\n',i,100*obj.aps(i)) ;
        end
      else
        for i=1:numel(obj.aps)
          fprintf('%-50s %.1f\n',obj.classNames{i},100*obj.aps(i)) ;
        end
      end
    end
    
    function aps = compute_average_precision(obj)
      assert(all(size(obj.scores)==size(obj.gtLabels)));
      % nImg = size(obj.scores,1);
      nCls = numel(obj.cls_index);

      aps = zeros(1,nCls);

      for c=1:nCls
        gt = obj.gtLabels(c,:);
        conf = obj.scores(c,:) ;
        if sum(gt>0)==0, continue ; end
        
        % compute average precision
        if obj.voc07
          [rec,prec,ap]=obj.VOC07ap(conf,gt) ;
        else
          [rec,prec,ap]=obj.THUMOSeventclspr(conf,gt) ;
        end
        aps(c) = ap;
      end
    end

    function [rec,prec,ap]=VOC07ap(obj,conf,gt)
      [~,si]=sort(-conf);
      tp=gt(si)>0;
      fp=gt(si)<0;
      
      fp=cumsum(fp);
      tp=cumsum(tp);
      
      rec=tp/sum(gt>0);
      prec=tp./(fp+tp);
      ap=0;
      for t=0:0.1:1
        p=max(prec(rec>=t));
        if isempty(p)
          p=0;
        end
        ap=ap+p/11;
      end
    end
    
    function [rec,prec,ap]=THUMOSeventclspr(obj,conf,gt)
      [so,sortind]=sort(-conf);
      tp=gt(sortind)==1;
      fp=gt(sortind)~=1;
      npos=length(find(gt==1));
      
      % compute precision/recall
      fp=cumsum(fp);
      tp=cumsum(tp);
      rec=tp/npos;
      prec=tp./(fp+tp);
      
      % compute average precision
      
      ap=0;
      tmp=gt(sortind)==1;
      for i=1:length(conf)
        if tmp(i)==1
          ap=ap+prec(i);
        end
      end
      ap=ap/npos;
    end
    
    function obj = LayerAP(varargin)
      obj.load(varargin) ;
      obj.loss = 'average_precision' ;
    end
  end
end
