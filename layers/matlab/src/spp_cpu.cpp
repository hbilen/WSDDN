// @file spp_cpu.cpp
// @brief Pooling block implementation (CPU)
// @author Hakan Bilen

#include "matrix.h"
#include "bits/data.hpp"
#include "spp.hpp"

#include <algorithm>
#include <limits>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <mex.h>

#ifndef VERBOSE
#define VERBOSE 0
#endif

using std::max;
using std::min;
/* ---------------------------------------------------------------- */
/*                                               Max spp helper */
/* ---------------------------------------------------------------- */

template <typename type>
struct acc_max
{
  inline acc_max(int poolHeight, int poolWidth, type derOutput = 0)
    :
      value(-std::numeric_limits<type>::infinity()),
      derOutput(derOutput),
      derDataActivePt(NULL)
  { }

  inline void accumulate_forward(type x) {
    value = std::max(value, x) ;
  }

  inline void accumulate_backward(type const* data, type* derDataPt) {
    type x = *data ;
    if (x > value) {
      value = x ;
      derDataActivePt = derDataPt ;
    }
  }

  inline type done_forward() const {
    return value ;
  }

  inline void done_backward() const {
    if (derDataActivePt) { *derDataActivePt += derOutput ; }
  }

  type value ;
  type derOutput ;
  type* derDataActivePt ;
} ;

/* ---------------------------------------------------------------- */
/*                                           Average pooling helper */
/* ---------------------------------------------------------------- */
template <typename type>
struct acc_sum
{
  inline acc_sum(int poolHeight, int poolWidth, type derOutput = 0)
  :
  value(0),
  scale(type(1)/type(poolHeight*poolWidth)),
  derOutput(derOutput)
  { }

  inline void accumulate_forward(type x) {
    value += x ;
  }

  inline void accumulate_backward(type const* data, type* derDataPt) {
    *derDataPt += derOutput * scale ;
  }

  inline type done_forward() const {
    return value * scale ;
  }

  inline void done_backward() const { }

  type value ;
  type derOutput ;
  type scale;
} ;


/* ---------------------------------------------------------------- */
/*                                                spp_*_forward */
/* ---------------------------------------------------------------- */

template<typename type, typename Accumulator> static inline void
spp_forward_cpu(type* pooled,
                type const* data,
                size_t height, size_t width, size_t depth, size_t size,
                size_t numTotBins,
                size_t numLevels, const float * levels,
                size_t numROIs, const float * ROIs )
{
  
  mxAssert(numLevels>0,"");
  mxAssert(numROIs>0,"");

  // For each ROI R = [level x1 y1 x2 y2]: max pool over R
  for (int roi = 0; roi < numROIs; ++roi) {
    int roi_image   = static_cast<int>(ROIs[5 * roi + 0]);
    int roi_start_h = static_cast<int>(ROIs[5 * roi + 1]);
    int roi_start_w = static_cast<int>(ROIs[5 * roi + 2]);
    int roi_end_h   = static_cast<int>(ROIs[5 * roi + 3]);
    int roi_end_w   = static_cast<int>(ROIs[5 * roi + 4]);
    if(roi_start_w==roi_end_w) {
      if(roi_start_w>0)
        roi_start_w--;
      else
        roi_end_w++;
    }
    if(roi_start_h==roi_end_h) {
      if(roi_start_h>0)
        roi_start_h--;
      else
        roi_end_h++;
    }
  
#if VERBOSE
    mexPrintf("%d : dim (%d %d %d %d) roi_image %d pool-roi [%d %d %d %d]\n",roi,width,height,depth,size,roi_image,roi_start_w,roi_start_h,roi_end_w,roi_end_h);
#endif

    mxAssert(roi_image<size && roi_image >= 0 ,"Invalid ROI image index.");
    mxAssert(roi_start_w >= 0 && roi_start_w < width, "Invalid ROI start_w");
    mxAssert(roi_start_h >= 0 && roi_start_h < height, "Invalid ROI start_h");
    mxAssert(roi_end_w >= 0 && roi_end_w < width, "Invalid ROI end w");
    mxAssert(roi_end_h >= 0 && roi_end_h < height,"Invalid ROI end h");
    mxAssert(roi_start_w <= roi_end_w, "Error ROI start w > ROI end w");
    mxAssert(roi_start_h <= roi_end_h, "Error ROI start h > ROI end h");


    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);


    type const * data_offset = data + (roi_image * depth) * (width*height);

    for (int z = 0; z < depth; ++z) {
      
      for (int l = 0; l < numLevels ; l++) {
        
        mxAssert(levels[l]>0,"");
        
        int pooledWidth  = static_cast<int>(levels[l]);
        int pooledHeight = static_cast<int>(levels[l]);
        
        const float bin_size_h = static_cast<float>(roi_height)
            / static_cast<float>(pooledHeight);
        const float bin_size_w = static_cast<float>(roi_width)
            / static_cast<float>(pooledWidth);
#if VERBOSE
       printf("width %d height %d pooledWidth %d pooledHeight %d bin_size_w %f bin_size_h %f\n",width,height,pooledWidth,pooledHeight,bin_size_w,bin_size_h);
#endif

        for (int ph = 0; ph < pooledHeight; ++ph) {
          for (int pw = 0; pw < pooledWidth; ++pw) {
            
            int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                                * bin_size_h));
            int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                                * bin_size_w));
            int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                             * bin_size_h));
            int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                             * bin_size_w));

            hstart = min(max(hstart + roi_start_h, 0), (int)height);
            hend = min(max(hend + roi_start_h, 0), (int)height);
            wstart = min(max(wstart + roi_start_w, 0), (int)width);
            wend = min(max(wend + roi_start_w, 0), (int)width);

#if VERBOSE
            printf("%d : level %d roi-image %d ph %d pw %d roi [%d %d %d %d]\n",roi,l,roi_image,ph,pw,wstart,hstart,wend,hend);
#endif
            mxAssert(hend>hstart,"");
            mxAssert(wend>wstart,"");
            
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            //            const int pool_index = ph * pooled_width_ + pw;
            if (is_empty) {
              pooled[pw * pooledHeight + ph] = 0;
              //              top_data[pool_index] = 0;
              //              argmax_data[pool_index] = -1;
            }

            Accumulator acc(hend - hstart, wend - wstart) ;
            for (int w = wstart; w < wend; ++w) {
              for (int h = hstart; h < hend; ++h) {
                const int index = w * height + h;
                acc.accumulate_forward(data_offset[index]) ;

#if VERBOSE
                printf("h %d w %d %f\n",h,w,data_offset[index]);
#endif
              }
            }

            pooled[pw * pooledHeight + ph] = acc.done_forward() ;
          } // end of pw
        } // end of ph
        pooled += pooledWidth*pooledHeight ;
        data_offset += width*height;
      } // end of pl
    } // end of z
  } // end of n
}

template<> vl::Error
vl::impl::spp_avg_forward<vl::CPU, float>(float* pooled,
                                          float const* data,
                                          size_t height, size_t width, size_t depth, size_t size,
                                          size_t numTotBins,
                                          size_t numLevels, const float * levels,
                                          size_t numROIs, const float * ROIs)
{

  // For each ROI R = [level x1 y1 x2 y2]: max pool over R
  for (size_t n = 0; n < numROIs; ++n) {

    int roi_image   = static_cast<int>(ROIs[5 * n + 0]);
    int roi_start_h = static_cast<int>(ROIs[5 * n + 1]);
    int roi_start_w = static_cast<int>(ROIs[5 * n + 2]);
    int roi_end_h   = static_cast<int>(ROIs[5 * n + 3]);
    int roi_end_w   = static_cast<int>(ROIs[5 * n + 4]);

    if(roi_start_w==roi_end_w) {
      if(roi_start_w>0)
        roi_start_w--;
      else
        roi_end_w++;
    }
    if(roi_start_h==roi_end_h) {
      if(roi_start_h>0)
        roi_start_h--;
      else
        roi_end_h++;
    }
 
#if VERBOSE
    mexPrintf("%d : pool-roi [%d %d %d %d]\n",n,roi_start_w,roi_start_h,roi_end_w,roi_end_h);
#endif
  }
  spp_forward_cpu<float, acc_sum<float> > (pooled,
                                           data,
                                           height, width, depth, size,
                                           numTotBins,
                                           numLevels, levels,
                                           numROIs, ROIs) ;
  return vlSuccess ;
}

template<> vl::Error
vl::impl::spp_max_forward<vl::CPU, float>(float* pooled,
                                          float const* data,
                                          size_t height, size_t width, size_t depth, size_t size,
                                          size_t numTotBins,
                                          size_t numLevels, const float * levels,
                                          size_t numROIs, const float * ROIs)
{

  // For each ROI R = [level x1 y1 x2 y2]: max pool over R
  for (size_t n = 0; n < numROIs; ++n) {

    int roi_image   = static_cast<int>(ROIs[5 * n + 0]);
    int roi_start_h = static_cast<int>(ROIs[5 * n + 1]);
    int roi_start_w = static_cast<int>(ROIs[5 * n + 2]);
    int roi_end_h   = static_cast<int>(ROIs[5 * n + 3]);
    int roi_end_w   = static_cast<int>(ROIs[5 * n + 4]);

    if(roi_start_w==roi_end_w) {
      if(roi_start_w>0)
        roi_start_w--;
      else
        roi_end_w++;
    }
    if(roi_start_h==roi_end_h) {
      if(roi_start_h>0)
        roi_start_h--;
      else
        roi_end_h++;
    }
 
#if VERBOSE
    mexPrintf("%d : pool-roi [%d %d %d %d]\n",n,roi_start_w,roi_start_h,roi_end_w,roi_end_h);
#endif
  }
  spp_forward_cpu<float, acc_max<float> > (pooled,
                                           data,
                                           height, width, depth, size,
                                           numTotBins,
                                           numLevels, levels,
                                           numROIs, ROIs) ;
  return vlSuccess ;
}



/* ---------------------------------------------------------------- */
/*                                               spp_*_backward */
/* ---------------------------------------------------------------- */

/*
 assume the output array to be cleared or otherwise
 properly initialised: accumulates the derivative
 */

/* Todo: transpose */

template<typename type, typename Accumulator> static inline void
spp_backward_cpu(type* derData,
                 type const* data,
                 type const* derPooled,
                 size_t height, size_t width, size_t depth, size_t size,
                 size_t numLevels, const float * levels,
                 size_t numROIs, const float * ROIs )
{

  mxAssert(numLevels>0,"");
  mxAssert(numROIs>0,"");
  for (int l = 0; l < numLevels ; l++) {
    mxAssert(levels[l]>0,"levels must be positive");
  }
#if VERBOSE
    mexPrintf("numROIs %d numLevels %d height %d width %d depth %d size %d\n",numROIs,numLevels,width,height,depth,size);
#endif


  // For each ROI R = [level x1 y1 x2 y2]: max pool over R
  for (size_t n = 0; n < numROIs; ++n) {

    int roi_image   = static_cast<int>(ROIs[5 * n + 0]);
    int roi_start_h = static_cast<int>(ROIs[5 * n + 1]);
    int roi_start_w = static_cast<int>(ROIs[5 * n + 2]);
    int roi_end_h   = static_cast<int>(ROIs[5 * n + 3]);
    int roi_end_w   = static_cast<int>(ROIs[5 * n + 4]);
    
    mxAssert(roi_image<size,"");
    mxAssert(roi_start_w>=0,"");
    mxAssert(roi_start_h>=0,"");
    mxAssert(roi_end_w<width,"");
    mxAssert(roi_end_h<height,"");

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);



    type const * data_offset = data + (roi_image * depth) * (width*height);
    type * derData_offset = derData + (roi_image * depth) * (width*height);

    for (int z = 0; z < depth; ++z) {
      for (int l = 0; l < numLevels ; l++) {
        
        int pooledWidth  = static_cast<int>(levels[l]);
        int pooledHeight = pooledWidth;
        
        const type bin_size_h = static_cast<type>(roi_height)
            / static_cast<type>(pooledHeight);
        const type bin_size_w = static_cast<type>(roi_width)
            / static_cast<type>(pooledWidth);


        for (int ph = 0; ph < pooledHeight; ++ph) {
          for (int pw = 0; pw < pooledWidth; ++pw) {
            
            int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                                * bin_size_h));
            int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                                * bin_size_w));
            int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                             * bin_size_h));
            int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                             * bin_size_w));

            hstart = min(max(hstart + roi_start_h, 0), (int)height);
            hend = min(max(hend + roi_start_h, 0), (int)height);
            wstart = min(max(wstart + roi_start_w, 0), (int)width);
            wend = min(max(wend + roi_start_w, 0), (int)width);

            //            bool is_empty = (hend <= hstart) || (wend <= wstart);
            //
            //            if (is_empty) {
            //              derPooled[ph * pooledWidth + pw] = 0;
            //            }

            Accumulator acc(hend - hstart, wend - wstart, derPooled[pw * pooledHeight + ph]) ;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = w * height + h ; 
                acc.accumulate_backward(&data_offset[index],
                                        &derData_offset[index]) ;

              }
            }
            acc.done_backward() ;
          } // end of pw
        } // end of ph
        data_offset += width*height ;
        derData_offset += width*height ;
        derPooled += pooledWidth*pooledHeight ;
      } // end of l
    } // end of z
  } // end of n
}

template<> vl::Error
vl::impl::spp_avg_backward<vl::CPU, float>(float* derData,
                                           float const* data,
                                           float const* derPooled,
                                           size_t height, size_t width, size_t depth, size_t size,
                                           size_t numTotBins,
                                           size_t numLevels, const float * levels,
                                           size_t numROIs, const float * ROIs)
{
  spp_backward_cpu<float, acc_sum<float> > (derData, data, derPooled,
                                            height, width, depth, size,
                                            numLevels, levels,
                                            numROIs, ROIs );
  return vlSuccess ;
}

template<> vl::Error
vl::impl::spp_max_backward<vl::CPU, float>(float* derData,
                                           float const* data,
                                           float const* derPooled,
                                           size_t height, size_t width, size_t depth, size_t size,
                                           size_t numTotBins,
                                           size_t numLevels, const float * levels,
                                           size_t numROIs, const float * ROIs)
{
  spp_backward_cpu<float, acc_max<float> > (derData, data, derPooled,
                                            height, width, depth, size,
                                            numLevels, levels,
                                            numROIs, ROIs );
  return vlSuccess ;
}
