classdef BiasSamples < dagnn.ElementWise
  % @author: Hakan Bilen
  properties
    scale = single(1)
  end
  properties (Transient)
    boxCoefs = []
  end
  methods
    function outputs = forward(obj, inputs, params)
      if numel(inputs) ~= 2
        error('Number of inputs is not 2');
      end
      obj.boxCoefs = single(1)+obj.scale*inputs{2};
      outputs{1} = bsxfun(@times,inputs{1},obj.boxCoefs);
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1,2) ;
      obj.boxCoefs = single(1)+obj.scale*inputs{2};
      derInputs{1} = bsxfun(@times,derOutputs{1},obj.boxCoefs) ;
      derParams = {} ;
    end
    
    function obj = BiasSamples(varargin)
      obj.load(varargin) ;
    end
    
    function reset(obj)
      obj.boxCoefs = [] ;
    end
    
  end
  
end
