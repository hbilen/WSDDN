classdef SPP < dagnn.Layer
  % @author: Hakan Bilen
  % Spatial Pyramid Pooling Layer (with backpropagation)
  
  % first proposed in
  % He, K., Zhang, X., Ren, S., & Sun, J.
  % "Spatial pyramid pooling in deep convolutional networks for visual recognition."
  % TPAMI (2015).
  
  % this implementations based on 
  % Girshick, R. "Fast R-CNN". ICCV (2015).
  
 
  properties
    levels = 6 % Levels - number of bins per side
    % levels = single([1 2 3 6]); % for multi-scale
    
    % parameters for transforming pixel to feature map coordinates
    fix_bounds = false
    stride = 16 % downscaling
    offset1 = 1 % first offset
    offset2 = 0 % second offset
    method = 'max' % 'max', 'avg' pooling
    
    num_bins % number of bins in spatial pyramid (sum(levels.^2))
  end
  
  properties (Transient)
    ROIs = [];
  end
  
  methods
    function nboxes = spm_response_boxes(self, boxes, w, h)
      o0 = self.offset1;
      o  = self.offset2;
      ss = self.stride;
      if numel(ss)==1
        ss(2) = ss(1);
      end
      
      nboxes = [ ...
        floor((boxes(1,:) - o0 + o) / ss(1) + 0.5);
        floor((boxes(2,:) - o0 + o) / ss(2) + 0.5);
        ceil((boxes(3,:) - o0 - o) / ss(1) - 0.5);
        ceil((boxes(4,:) - o0 - o) / ss(2) - 0.5)];
      
      function a = fix_invalid(a)
        inval = a(1,:) > a(2,:);
        a(1,inval) = floor((a(1,inval) + a(2,inval))./2);
        a(2,inval) = a(1,inval);
      end
      
      nboxes([1 3],:) = fix_invalid(nboxes([1 3],:));
      nboxes([2 4],:) = fix_invalid(nboxes([2 4],:));
      
      nboxes = [...
        min(h-2, max(nboxes(1,:), 0));
        min(w-2, max(nboxes(2,:), 0));
        min(h-1, max(nboxes(3,:), 0));
        min(w-1, max(nboxes(4,:), 0))];
      
    end
    
    function self = SPP(varargin)
      self.load(varargin) ;
      self.levels = single(self.levels);
      self.num_bins = double(gather(sum(self.levels.^2)));
      self.stride = self.stride;
      self.offset1 = self.offset1;
      self.offset2 = self.offset2;
    end
    
    function outputs = forward(self, inputs, params )
      
      [h, w, ~, n] = size(inputs{1});
      if numel(inputs)==1
        rois = [1 1 h w]';
      else
        rois = inputs{2};
      end
      boxes = self.spm_response_boxes(rois(2:5,:), w, h);
      rois(1,:) = rois(1,:) - 1;
      if max(rois(1,:))>=n || any(rois(1,:))<0
        error('wrong roi');
      end
      rois(2:5,:) = single(boxes);
      
      if isa(inputs{1},'gpuArray')
        if ~isa(self.levels,'gpuArray')
          self.levels = gpuArray(self.levels);
        end
        if ~isa(rois,'gpuArray')
          rois = gpuArray(rois);
        end
      end
      self.ROIs = rois;
      outputs{1} = vl_nnspp(inputs{1}, self.levels, rois, ...
        'numbins', self.num_bins, 'method', self.method)  ;
    end
    
    
    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      [h, w, d, n] = size(inputs{1});

      if isempty(self.ROIs)
        if numel(inputs)==1
          rois = [1 1 h w]';
        else
          rois = inputs{2};
        end
        
        boxes = self.spm_response_boxes(rois(2:5,:), w, h);
        rois(1,:) = rois(1,:) - 1;
        rois(2:5,:) = single(boxes);
        if max(rois(1,:))>=n || any(rois(1,:))<0
          error('wrong roi');
        end
        
        if isa(inputs{1},'gpuArray')
          %         if ~isa(self.levels,'gpuArray')
          %           self.levels = gpuArray(self.levels);
          %         end
          if ~isa(rois,'gpuArray')
            rois = gpuArray(rois);
          end
        end
        self.ROIs = rois;
      end
      
      if numel(size(inputs{1}))==3
        inputs{1} = reshape(inputs{1},[size(inputs{1}) 1]);
        derOutputs{1} = reshape(derOutputs{1},[size(derOutputs{1}) 1]);
      end
      derInputs{1} = vl_nnspp(inputs{1}, self.levels, self.ROIs, derOutputs{1}, ...
        'numbins', self.num_bins, 'method', self.method) ;
      
      derInputs{2} = [];
      derParams = {} ;
    end
    
    function reset(obj)      
      obj.levels = gather(obj.levels) ;
    end
  end
end

