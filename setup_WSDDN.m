function setup_WSDDN()
%SETUP_WSDDN Sets up WSDDN, by adding its folders to the Matlab path

root = fileparts(mfilename('fullpath')) ;
addpath(root, [root '/matlab'], [root '/pascal'], [root '/core']) ;
%addpath([vl_rootnn '/matlab/mex']) ; % tmp fix for compiler
addpath([vl_rootnn '/examples/fast_rcnn/bbox_functions']) ;
addpath([root '/imagenet']) ;
