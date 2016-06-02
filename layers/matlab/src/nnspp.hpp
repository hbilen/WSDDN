// @file nnspp.hpp
// @brief Pooling block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnspp__
#define __vl__nnspp__

#include "bits/data.hpp"
#include <stdio.h>

namespace vl {

vl::Error
nnspp_forward(vl::Context& context,
              vl::Tensor output,
              vl::Tensor data,
              size_t method,
              size_t numTotBins,
              vl::Tensor levels,
              vl::Tensor ROIs ) ;

vl::Error
nnspp_backward(vl::Context& context,
               vl::Tensor derData,
               vl::Tensor data,
               vl::Tensor derOutput,
               size_t method,
               size_t numTotBins,
               vl::Tensor levels,
               vl::Tensor ROIs ) ;
}

#endif /* defined(__vl__nnspp__) */
