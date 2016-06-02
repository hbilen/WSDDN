classdef Times < dagnn.ElementWise
  % @author: Hakan Bilen
  % Times (multiply) DagNN layer
  %   The Times layer takes the multiplication of two inputs and store the result
  %   as its only output.
  methods
    function outputs = forward(obj, inputs, params)
      if numel(inputs) ~= 2
        error('Number of inputs is not 2');
      end
      outputs{1} = inputs{1} .* inputs{2} ;
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1,2) ;
      derInputs{1} = derOutputs{1} .* inputs{2}  ;
      derInputs{2} = derOutputs{1} .* inputs{1}  ;
      derParams = {} ;
    end
    
    function obj = Times(varargin)
      obj.load(varargin) ;
    end
    
  end
  
end