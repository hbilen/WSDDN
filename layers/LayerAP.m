classdef LayerAP < dagnn.Loss
  % @author: Hakan Bilen
  % 11 step average precision
  properties (Transient)
    gtLabels = []
    scores   = []
    ids      = []
%     average = 0;
    aps = 0;
  end

  methods
    function outputs = forward(obj, inputs, params)
      if numel(inputs)==2
      
        obj.scores = [obj.scores;gather(squeeze(inputs{1}))'];
        obj.gtLabels = [obj.gtLabels;gather(squeeze(inputs{2}))'];
      elseif numel(inputs)>2
        scoresCur = gather(squeeze(inputs{1}))';
        gtLabelsCur = gather(squeeze(inputs{2}))';
        idsCur = gather(squeeze(inputs{3}))';
        
        [lia,locb] = ismember(idsCur,obj.ids);
        
        if any(lia)
          obj.scores = [obj.scores;scoresCur(~lia,:)];
          obj.gtLabels = [obj.gtLabels;gtLabelsCur(~lia,:)];
          obj.ids = [obj.ids;idsCur(~lia,:)];
          
          nz = find(lia);
          for i=1:numel(nz)
            obj.scores(locb(nz(i)),:) = obj.scores(locb(nz(i)),:) + ...
              scoresCur(nz(i),:);
          end
        else
          obj.scores = [obj.scores;scoresCur];
          obj.gtLabels = [obj.gtLabels;gtLabelsCur];
          obj.ids = [obj.ids;idsCur];
        end
      else
        error('wrong number of inputs');
      end
      
      obj.aps = obj.compute_average_precision();
      obj.average = 100 * mean(obj.aps);
      outputs{1} =  100 * mean(obj.aps);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1,3);
      derInputs{1} = derOutputs{1} ;
      derParams = {} ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.aps = 0 ;
      obj.gtLabels = [];
      obj.scores   = [];
      obj.ids      = [];
    end


    
    function aps = compute_average_precision(obj)
      assert(all(size(obj.scores)==size(obj.gtLabels)));
      % nImg = size(obj.scores,1);
      nCls = size(obj.scores,2);

      aps = zeros(1,nCls);

      for c=1:nCls
        gt = obj.gtLabels(:,c);
        [~,si]=sort(-obj.scores(:,c));
        tp=gt(si)>0;
        fp=gt(si)<0;
        
        fp=cumsum(fp);
        tp=cumsum(tp);
%         gt(gt==0) = 1e-8;
%         fp(fp==0) = 1e-8;
        rec=tp/sum(gt>0);
        prec=tp./(fp+tp);
        
        % compute average precision
        
        ap=0;
        for t=0:0.1:1
          p=max(prec(rec>=t));
          if isempty(p)
            p=0;
          end
          ap=ap+p/11;
        end
        aps(c) = ap;
      end
    end

    function obj = LayerAP(varargin)
      obj.load(varargin) ;
    end
  end
end
