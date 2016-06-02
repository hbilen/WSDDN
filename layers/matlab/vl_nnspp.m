% VL_NNSPP  CNN spatial pyramid pooling

% @inproceedings{kaiming14ECCV,
%     Author = {Kaiming, He and Xiangyu, Zhang and Shaoqing, Ren and Jian Sun},
%     Title = {Spatial pyramid pooling in deep convolutional networks for visual recognition},
%     Booktitle = {European Conference on Computer Vision},
%     Year = {2014}
% }
%    Y = VL_NNSPP(X, PLEVELS, ROIS, 'NUMBINS', NB) 
%    applies the max spatial pyramid pooling operator to region of interest
%    ROIS of channels of the data X. 
%
%    X is a SINGLE array of dimension H x W x D x N where (H,W) are the
%    height and width of the map stack, D is the image depth (number
%    of feature channels) and N the number of of images in the stack.
%
%    ROIS 5xK single array that needs to be loaded to gpu in case of
%    gpu use. K is the number of ROIs, each column of ROIS has [ID, X0, Y0, X1, Y1]
%    where ID is between 1-N (number of images). X0, Y0, X1, Y1 correspond
%    to coordinates of the ROI in feature map X. Thus pixels coordinates
%    has to be converted into feature map coordinates.
%
%    PLEVELS is an array of pyramid levels. For example, [2] corresponds 2x2
%    spatial pooling with 4 bins (NUMBINS). [1,2,3,4] means 1x1+2x2+3x3+4x4
%    = 31 bins. 
%
%    NB is number of total bins. sum(PLEVELS(:)) 
%
%    DZDX = VL_NNSPP(X, PLEVELS, ROIS, 'NUMBINS', NB, DZDY) computes the derivatives of
%    the nework output Z w.r.t. the data X given the derivative DZDY
%    w.r.t the max-pooling output Y.
%
%    VL_NNCONV(..., 'option', value, ...) takes the following options:
%
% Copyright (C) 2014 Andrea Vedaldi, Karel Lenc, and Max Jaderberg.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

