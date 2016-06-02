// @file spp_gpu.cu
// @brief  SPP block implementation (GPU)
// @author Hakan Bilen

#include "spp.hpp"
#include "bits/datamex.hpp"
#include "bits/datacu.hpp"
#include "matrix.h"

#include <float.h>
#include <sm_20_atomic_functions.h>
#include <cmath>
#include <stdio.h>

/* ---------------------------------------------------------------- */
/*                                              spp_max_forward */
/* ---------------------------------------------------------------- */
template<typename T> __global__ void
spp_avg_kernel
(T* pooled,
 const T* data,
 const int height,
 const int width,
 const int depth,
 const int size,
 const int numTotBins,
 const int numLevels,
 const float * levels,
 const int numROIs,
 const float * ROIs)
{
  mxAssert(numLevels>0,"");
  mxAssert(numROIs>0,"");
  mxAssert(numTotBins>0,"");


  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;


  int pooledVolume = numTotBins * depth * numROIs;

  if (pooledIndex < pooledVolume) {

    int pl = pooledIndex % numTotBins;
    int pc = (pooledIndex / numTotBins) % depth;
    int pr = (pooledIndex / numTotBins / depth); // roi no

    mxAssert(pr<numROIs,"");

    int roi_image   = ROIs[5 * pr + 0];
    mxAssert(roi_image<size,"");

    int roi_start_h = ROIs[5 * pr + 1];
    int roi_start_w = ROIs[5 * pr + 2];
    int roi_end_h   = ROIs[5 * pr + 3];
    int roi_end_w   = ROIs[5 * pr + 4];

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

    mxAssert(roi_start_w>=0,"");
    mxAssert(roi_start_h>=0,"");
    mxAssert(roi_end_w<width,"");
    mxAssert(roi_end_h<height,"");

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    // Find pyramid level and bin
    int pb = -1;
    int pLevel = -1;
    int numBins = 0;
    for(int l=0;l<numLevels;l++) {
      if(pl-numBins>=0 && pl-numBins<static_cast<int>(levels[l] * levels[l])) {
        pb = pl - numBins;
        pLevel = l;
      }
      numBins += static_cast<int>(levels[l] * levels[l]);
    }
    
    mxAssert(pb>=0,"");
    mxAssert(pLevel>=0,"");
    int pooledWidth  = levels[pLevel];
    int pooledHeight = levels[pLevel];
    int pw = pb / pooledHeight;
    int ph = pb % pooledHeight;


    const float bin_size_h = static_cast<float>(roi_height)
        / static_cast<float>(pooledHeight);
    const float bin_size_w = static_cast<float>(roi_width)
        / static_cast<float>(pooledWidth);


    mxAssert(ph>-1 && pw>-1,"");

    int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);

    int offset_data = (roi_image * depth + pc) * (width*height);
    mxAssert(offset_data<width*height*depth*size,"");

    data += offset_data;
    T bestValue = 0;
    const float coef = 1.f / (float)((wend-wstart) * (hend-hstart));
    for (int w = wstart; w < wend; ++w) {
      for (int h = hstart; h < hend; ++h) {
        int index = w * height + h ;
        bestValue += data[index] * coef;
      }
    }
    pooled[pooledIndex] = bestValue ;
  }
}

template<typename T> __global__ void
spp_max_kernel
(T* pooled,
 const T* data,
 const int height,
 const int width,
 const int depth,
 const int size,
 const int numTotBins,
 const int numLevels,
 const float * levels,
 const int numROIs,
 const float * ROIs)
{
  mxAssert(numLevels>0,"");
  mxAssert(numROIs>0,"");
  mxAssert(numTotBins>0,"");


  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;


  int pooledVolume = numTotBins * depth * numROIs;

  if (pooledIndex < pooledVolume) {

    int pl = pooledIndex % numTotBins;
    int pc = (pooledIndex / numTotBins) % depth;
    int pr = (pooledIndex / numTotBins / depth); // roi no
    mxAssert(pr<numROIs,"");

    int roi_image   = ROIs[5 * pr + 0];
    mxAssert(roi_image<size,"");

    int roi_start_h = ROIs[5 * pr + 1];
    int roi_start_w = ROIs[5 * pr + 2];
    int roi_end_h   = ROIs[5 * pr + 3];
    int roi_end_w   = ROIs[5 * pr + 4];

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

    mxAssert(roi_start_w>=0,"");
    mxAssert(roi_start_h>=0,"");
    mxAssert(roi_end_w<width,"");
    mxAssert(roi_end_h<height,"");

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    // Find pyramid level and bin
    int pb = -1;
    int pLevel = -1;
    int numBins = 0;
    for(int l=0;l<numLevels;l++) {
      if(pl-numBins>=0 && pl-numBins<static_cast<int>(levels[l] * levels[l])) {
        pb = pl - numBins;
        pLevel = l;
      }
      numBins += static_cast<int>(levels[l] * levels[l]);
    }
    
    mxAssert(pb>=0,"");
    mxAssert(pLevel>=0,"");
    int pooledWidth  = levels[pLevel];
    int pooledHeight = levels[pLevel];
    int pw = pb / pooledHeight;
    int ph = pb % pooledHeight;


    const float bin_size_h = static_cast<float>(roi_height)
        / static_cast<float>(pooledHeight);
    const float bin_size_w = static_cast<float>(roi_width)
        / static_cast<float>(pooledWidth);

//    free(numBins);

    mxAssert(ph>-1 && pw>-1,"");

    int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    int offset_data = (roi_image * depth + pc) * (width*height);
    mxAssert(offset_data<width*height*depth*size,"");

    data += offset_data;
    T bestValue = is_empty ? 0 : data[wstart * height + hstart];
    for (int w = wstart; w < wend; ++w) {
      for (int h = hstart; h < hend; ++h) {
        int index = w * height + h ;
        bestValue = max(bestValue, data[index]) ;
      }
    }
    pooled[pooledIndex] = bestValue ;

  }
}

template<> vl::Error
vl::impl::spp_avg_forward<vl::GPU, float>(float* pooled,
                                          float const* data,
                                          size_t height, size_t width, size_t depth, size_t size,
                                          size_t numTotBins,
                                          size_t numLevels, const float * levels,
                                          size_t numROIs, const float * ROIs)
{
  mxAssert(numLevels>0,"");
  mxAssert(numROIs>0,"");

  int pooledVolume = numTotBins * depth * numROIs;

  spp_avg_kernel<float><<< divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS),
      VL_CUDA_NUM_THREADS >>>(pooled, data,
                              height, width, depth, size,
                              numTotBins,
                              numLevels, levels,
                              numROIs, ROIs);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}

template<> vl::Error
vl::impl::spp_max_forward<vl::GPU, float>(float* pooled,
                                          float const* data,
                                          size_t height, size_t width, size_t depth, size_t size,
                                          size_t numTotBins,
                                          size_t numLevels, const float * levels,
                                          size_t numROIs, const float * ROIs)
{
  mxAssert(numLevels>0,"");
  mxAssert(numROIs>0,"");

  int pooledVolume = numTotBins * depth * numROIs;

  spp_max_kernel<float><<< divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS),
      VL_CUDA_NUM_THREADS >>>(pooled, data,
                              height, width, depth, size,
                              numTotBins,
                              numLevels, levels,
                              numROIs, ROIs);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}
/* ---------------------------------------------------------------- */
/*                                              spp_max_backward */
/* ---------------------------------------------------------------- */
template<typename T> __global__ void
spp_max_backward_kernel
(T* derData,
 const T* data,
 const T* derPooled,
 const int height,
 const int width,
 const int depth,
 const int size,
 const int numTotBins,
 const int numLevels,
 const float * levels,
 const int numROIs,
 const float * ROIs)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;

  mxAssert(numLevels>0,"");
  mxAssert(numROIs>0,"");

  int pooledVolume = numTotBins * depth * numROIs;

  if (pooledIndex < pooledVolume) {


    int pl = pooledIndex % numTotBins;
    int pc = (pooledIndex / numTotBins) % depth;
    int pr = (pooledIndex / numTotBins / depth); // roi no
    mxAssert(pr<numROIs,"");

    int roi_image   = ROIs[5 * pr + 0];
    mxAssert(roi_image<size,"");

    int roi_start_h = ROIs[5 * pr + 1];
    int roi_start_w = ROIs[5 * pr + 2];
    int roi_end_h   = ROIs[5 * pr + 3];
    int roi_end_w   = ROIs[5 * pr + 4];

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
    mxAssert(roi_start_w>=0,"");
    mxAssert(roi_start_h>=0,"");
    mxAssert(roi_end_w<width,"");
    mxAssert(roi_end_h<height,"");

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    // Find pyramid level and bin
    int pb = -1;
    int pLevel = -1;
    int numBins = 0;
    for(int l=0;l<numLevels;l++) {
      if(pl-numBins>=0 && pl-numBins<static_cast<int>(levels[l] * levels[l])) {
        pb = pl - numBins;
        pLevel = l;
      }
      numBins += static_cast<int>(levels[l] * levels[l]);
    }
    
    mxAssert(pb>=0,"");
    mxAssert(pLevel>=0,"");
    int pooledWidth  = levels[pLevel];
    int pooledHeight = levels[pLevel];
    int pw = pb / pooledHeight;
    int ph = pb % pooledHeight;


    const float bin_size_h = static_cast<float>(roi_height)
        / static_cast<float>(pooledHeight);
    const float bin_size_w = static_cast<float>(roi_width)
        / static_cast<float>(pooledWidth);


    mxAssert(ph>-1 && pw>-1,"");

    int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    data += (roi_image * depth + pc) * (width*height);
    derData += (roi_image * depth + pc) * (width*height);

    int bestIndex = wstart * height + hstart;
    T bestValue = is_empty ? 0 : data[bestIndex];
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int index = w * height + h ;
        T value = data[index] ;
        if (value > bestValue) {
          bestValue = value ;
          bestIndex = index ;
        }
      }
    }

    /*
     This is bad, but required to eliminate a race condition when writing
     to bottom_diff.
     Caffe goes the other way around, but requires remembering the layer
     output, or the maximal indexes.
     atomicAdd(add, val)
     */
    atomicAdd(derData + bestIndex, derPooled[pooledIndex]) ;
  }
}


/* ---------------------------------------------------------------- */
/*                                              spp_avg_backward */
/* ---------------------------------------------------------------- */
template<typename T> __global__ void
spp_avg_backward_kernel
(T* derData,
 const T* data,
 const T* derPooled,
 const int height,
 const int width,
 const int depth,
 const int size,
 const int numTotBins,
 const int numLevels,
 const float * levels,
 const int numROIs,
 const float * ROIs)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;

  mxAssert(numLevels>0,"numLevels>0");
  mxAssert(numROIs>0,"numROIs>0");

  //  int numTotBins = 0;
  //  numTotBins = numBins[numLevels];

  int pooledVolume = numTotBins * depth * numROIs;

  if (pooledIndex < pooledVolume) {

    int pl = pooledIndex % numTotBins;
    int pc = (pooledIndex / numTotBins) % depth;
    int pr = (pooledIndex / numTotBins / depth); // roi no
    mxAssert(pr<numROIs,"");

    int roi_image   = ROIs[5 * pr + 0];
    mxAssert(roi_image<size,"");

    int roi_start_h = ROIs[5 * pr + 1];
    int roi_start_w = ROIs[5 * pr + 2];
    int roi_end_h   = ROIs[5 * pr + 3];
    int roi_end_w   = ROIs[5 * pr + 4];

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
    mxAssert(roi_start_w>=0,"");
    mxAssert(roi_start_h>=0,"");
    mxAssert(roi_end_w<width,"");
    mxAssert(roi_end_h<height,"");

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    // Find pyramid level and bin
    int pb = -1;
    int pLevel = -1;
    int numBins = 0;
    for(int l=0;l<numLevels;l++) {
      if(pl-numBins>=0 && pl-numBins<static_cast<int>(levels[l] * levels[l])) {
        pb = pl - numBins;
        pLevel = l;
      }
      numBins += static_cast<int>(levels[l] * levels[l]);
    }
    
    mxAssert(pb>=0,"");
    mxAssert(pLevel>=0,"");
    int pooledWidth  = levels[pLevel];
    int pooledHeight = levels[pLevel];
    int pw = pb / pooledHeight;
    int ph = pb % pooledHeight;


    const float bin_size_h = static_cast<float>(roi_height)
        / static_cast<float>(pooledHeight);
    const float bin_size_w = static_cast<float>(roi_width)
        / static_cast<float>(pooledWidth);


    mxAssert(ph>-1 && pw>-1,"");

    int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);

    data += (roi_image * depth + pc) * (width*height);
    derData += (roi_image * depth + pc) * (width*height);

    const float coef = 1.f / (float)((wend-wstart)*(hend-hstart));
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int index = w * height + h ;
      /*
       This is bad, but required to eliminate a race condition when writing
       to bottom_diff.
       Caffe goes the other way around, but requires remembering the layer
       output, or the maximal indexes.
       atomicAdd(add, val)
       */
        atomicAdd(derData + index, derPooled[pooledIndex] * coef) ;
      }
    }


  }
}

template<> vl::Error
vl::impl::spp_max_backward<vl::GPU, float>(float* derData,
                                           float const* data,
                                           float const* derPooled,
                                           size_t height, size_t width, size_t depth, size_t size,
                                           size_t numTotBins,
                                           size_t numLevels, const float * levels,
                                           size_t numROIs, const float * ROIs)
{
  mxAssert(numLevels>0,"");
  mxAssert(numROIs>0,"");

  int pooledVolume = numTotBins * depth * numROIs;

  spp_max_backward_kernel<float>
      <<< divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
                                                                                  (derData, data, derPooled,
                                                                                   height, width, depth, size,
                                                                                   numTotBins,
                                                                                   numLevels, levels,
                                                                                   numROIs, ROIs);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}

template<> vl::Error
vl::impl::spp_avg_backward<vl::GPU, float>(float* derData,
                                           float const* data,
                                           float const* derPooled,
                                           size_t height, size_t width, size_t depth, size_t size,
                                           size_t numTotBins,
                                           size_t numLevels, const float * levels,
                                           size_t numROIs, const float * ROIs)
{
  mxAssert(numLevels>0,"");
  mxAssert(numROIs>0,"");

  int pooledVolume = numTotBins * depth * numROIs;

  spp_avg_backward_kernel<float>
      <<< divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
                                                                                  (derData, data, derPooled,
                                                                                   height, width, depth, size,
                                                                                   numTotBins,
                                                                                   numLevels, levels,
                                                                                   numROIs, ROIs);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}
