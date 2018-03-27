classdef SumOverDim < dagnn.ElementWise
  % @author: Hakan Bilen
  % SumOverDim is the sum of the elements of inputs{1} over dimension dim
  properties 
    dim = 3;
  end
  
  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = sum(inputs{1},obj.dim) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      
      ndims = ones(1,numel(size(inputs{1})));
      ndims(obj.dim) = size(inputs{1},obj.dim); 
      derInputs{1} = repmat(derOutputs{1},ndims);
      
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
      outputSizes{1}(obj.dim) = 1;
    end

    function obj = SumOverDim(varargin)
      obj.load(varargin) ;
      obj.dim = obj.dim;
    end
  end
end
