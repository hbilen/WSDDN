// @file spp.hpp
// @brief SPP block implementation
// @author Hakan Bilen 

#ifndef VL_NNSPP_H
#define VL_NNSPP_H

#include "bits/data.hpp"
#include <cstddef>

namespace vl { namespace impl {


/* Average pooling */
template<vl::Device dev, typename type> vl::Error
spp_avg_forward(type* pooled,
                type const* data,
                size_t height, size_t width, size_t depth, size_t size,
                size_t numTotBins,
                size_t numLevels, const float * levels,
                size_t numROIs, const float * ROIs ) ;

template<vl::Device dev, typename type> vl::Error
spp_avg_backward(type* derData,
                 type const* data,
                 type const* derPooled,
                 size_t height, size_t width, size_t depth, size_t size,
                 size_t numTotBins,
                 size_t numLevels, const float * levels,
                 size_t numROIs, const float * ROIs) ;



/* Max pooling */
template<vl::Device dev, typename type> vl::Error
spp_max_forward(type* pooled,
                type const* data,
                size_t height, size_t width, size_t depth, size_t size,
                size_t numTotBins,
                size_t numLevels, const float * levels,
                size_t numROIs, const float * ROIs ) ;

template<vl::Device dev, typename type> vl::Error
spp_max_backward(type* derData,
                 type const* data,
                 type const* derPooled,
                 size_t height, size_t width, size_t depth, size_t size,
                 size_t numTotBins,
                 size_t numLevels, const float * levels,
                 size_t numROIs, const float * ROIs) ;


/* Specializations: CPU, float */
template<> vl::Error
spp_avg_forward<vl::CPU, float>(float* pooled,
                                float const* data,
                                size_t height, size_t width, size_t depth, size_t size,
                                size_t numTotBins,
                                size_t numLevels, const float * levels,
                                size_t numROIs, const float * ROIs ) ;


template<> vl::Error
spp_avg_backward<vl::CPU, float>(float* derData,
                                 float const* data,
                                 float const* derPooled,
                                 size_t height, size_t width, size_t depth, size_t size,
                                 size_t numTotBins,
                                 size_t numLevels, const float * levels,
                                 size_t numROIs, const float * ROIs ) ;


template<> vl::Error
spp_max_forward<vl::CPU, float>(float* pooled,
                                float const* data,
                                size_t height, size_t width, size_t depth, size_t size,
                                size_t numTotBins,
                                size_t numLevels, const float * levels,
                                size_t numROIs, const float * ROIs ) ;


template<> vl::Error
spp_max_backward<vl::CPU, float>(float* derData,
                                 float const* data,
                                 float const* derPooled,
                                 size_t height, size_t width, size_t depth, size_t size,
                                 size_t numTotBins,
                                 size_t numLevels, const float * levels,
                                 size_t numROIs, const float * ROIs ) ;


/* Specializations: GPU, float */

#if ENABLE_GPU
template<> vl::Error
spp_avg_forward<vl::GPU, float>(float* pooled,
                                float const* data,
                                size_t height, size_t width, size_t depth, size_t size,
                                size_t numTotBins,
                                size_t numLevels, const float * levels,
                                size_t numROIs, const float * ROIs ) ;

template<> vl::Error
spp_avg_backward<vl::GPU, float>(float* derData,
                                 float const* data,
                                 float const* derPooled,
                                 size_t height, size_t width, size_t depth, size_t size,
                                 size_t numTotBins,
                                 size_t numLevels, const float * levels,
                                 size_t numROIs, const float * ROIs ) ;
template<> vl::Error
spp_max_forward<vl::GPU, float>(float* pooled,
                                float const* data,
                                size_t height, size_t width, size_t depth, size_t size,
                                size_t numTotBins,
                                size_t numLevels, const float * levels,
                                size_t numROIs, const float * ROIs ) ;

template<> vl::Error
spp_max_backward<vl::GPU, float>(float* derData,
                                 float const* data,
                                 float const* derPooled,
                                 size_t height, size_t width, size_t depth, size_t size,
                                 size_t numTotBins,
                                 size_t numLevels, const float * levels,
                                 size_t numROIs, const float * ROIs ) ;


#endif

} }

#endif /* defined(VL_NNSPP_H) */
