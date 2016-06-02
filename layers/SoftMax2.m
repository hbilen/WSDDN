classdef SoftMax2 < dagnn.ElementWise
  % @author: Hakan Bilen
  % Softmax2 : it is a more generic softmax layer with a dimension and temperature parameter
  properties
    dim = 3;
    temp = 1;
    scale = 1;
  end
  
  methods
    function outputs = forward(self, inputs, params)
      inputs{1} = inputs{1} / self.temp;
      order = 1:numel(size(inputs{1}));
      if self.dim~=3
        order([3 self.dim]) = [self.dim 3];
        inputs{1} = permute(inputs{1},order);
      end
      outputs{1} = vl_nnsoftmax(inputs{1}) ;
      if self.dim~=3
        outputs{1} = permute(outputs{1},order) ;
      end
    end
    
    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      
      inputs{1} = inputs{1} / self.temp;
      order = 1:numel(size(inputs{1}));
      if self.dim~=3
        order(3) = self.dim;
        order(self.dim) = 3;
        inputs{1} = permute(inputs{1},order);
        derOutputs{1} = permute(derOutputs{1},order);
      end
      
      derInputs{1} = vl_nnsoftmax(inputs{1}, derOutputs{1}) ;
      if self.dim~=3
        derInputs{1} = permute(derInputs{1},order) ;
      end
      derParams = {} ;
    end
    
    function obj = SoftMax2(varargin)
      obj.load(varargin) ;
      obj.dim   = single(obj.dim);
      obj.temp  = single(obj.temp);
      obj.scale = single(obj.scale);
    end
  end
end

