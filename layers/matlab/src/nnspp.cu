// @file nnspp.cu
// @brief SPP block implementation (GPU)
// @author Hakan Bilen 


#include "nnspp.hpp"
#include "spp.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                nnspp_forward */
/* ---------------------------------------------------------------- */

Error
vl::nnspp_forward(vl::Context& context,
                  vl::Tensor output,
                  vl::Tensor data,
                  size_t method,
                  size_t numTotBins,
                  vl::Tensor levels,
                  vl::Tensor ROIs)
{
    Error status = vlSuccess ;
    switch (output.getDeviceType()) {
    default:
        assert(false) ;
        return vl::vlErrorUnknown ;

    case vl::CPU:
        if(!method) {
            status = vl::impl::spp_max_forward<CPU,float>
                    ((float*)output.getMemory(), (float const*)data.getMemory(),
                     data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(),
                     numTotBins,
                     levels.getNumElements(), (const float *)levels.getMemory(),
                     ROIs.getNumElements()/5, (const float *)ROIs.getMemory()) ;
        }
        else {
            status = vl::impl::spp_avg_forward<CPU,float>
                    ((float*)output.getMemory(), (float const*)data.getMemory(),
                     data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(),
                     numTotBins,
                     levels.getNumElements(), (const float *)levels.getMemory(),
                     ROIs.getNumElements()/5, (const float *)ROIs.getMemory()) ;
        }
        break ;

#ifdef ENABLE_GPU
    case vl::GPU:
        if(!method) {
            status = vl::impl::spp_max_forward<GPU,float>
                    ((float*)output.getMemory(), (float const*)data.getMemory(),
                     data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(),
                     numTotBins,
                     levels.getNumElements(), (float const *)levels.getMemory(),
                     ROIs.getNumElements()/5, (float const *)ROIs.getMemory()) ;
        }
        else {
            status = vl::impl::spp_avg_forward<GPU,float>
                    ((float*)output.getMemory(), (float const*)data.getMemory(),
                     data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(),
                     numTotBins,
                     levels.getNumElements(), (float const *)levels.getMemory(),
                     ROIs.getNumElements()/5, (float const *)ROIs.getMemory()) ;
        }
        if (status == vlErrorCuda) {
            context.setError(context.getCudaHelper().catchCudaError("spp_*_forward")) ;
        }
        break ;
#endif
    }
    return context.passError(status, "nnspp_forward: ") ;
}

/* ---------------------------------------------------------------- */
/*                                               nnspp_backward */
/* ---------------------------------------------------------------- */

Error
vl::nnspp_backward(Context& context,
                   Tensor derData,
                   Tensor data,
                   Tensor derPooled,
                   size_t method,
                   size_t numTotBins,
                   Tensor levels,
                   Tensor ROIs)
{
    vl::Error status = vlSuccess ;
    switch (derData.getDeviceType()) {
    default:
        assert(false) ;
        return vl::vlErrorUnknown ;

    case vl::CPU:
        if (!method) {
            status = vl::impl::spp_max_backward<CPU,float>
                    ((float*)derData.getMemory(), (float const*)data.getMemory(), (float const*)derPooled.getMemory(),
                     derData.getHeight(), derData.getWidth(), derData.getDepth(), derData.getSize(),
                     numTotBins,
                     levels.getNumElements(), (const float *)levels.getMemory(),
                     ROIs.getNumElements()/5, (const float *)ROIs.getMemory()) ;


        }
        else {
            status = vl::impl::spp_avg_backward<CPU,float>
                    ((float*)derData.getMemory(), (float const*)data.getMemory(), (float const*)derPooled.getMemory(),
                     derData.getHeight(), derData.getWidth(), derData.getDepth(), derData.getSize(),
                     numTotBins,
                     levels.getNumElements(), (const float *)levels.getMemory(),
                     ROIs.getNumElements()/5, (const float *)ROIs.getMemory()) ;
        }
        break ;



#if ENABLE_GPU
    case vl::GPU:
        if (!method) {
            status = vl::impl::spp_max_backward<GPU,float>
                    ((float*)derData.getMemory(), (float const*)data.getMemory(), (float const*)derPooled.getMemory(),
                     derData.getHeight(), derData.getWidth(), derData.getDepth(), derData.getSize(),
                     numTotBins,
                     levels.getNumElements(), (const float *)levels.getMemory(),
                     ROIs.getNumElements()/5, (const float *)ROIs.getMemory()) ;
        }
        else {
            status = vl::impl::spp_avg_backward<GPU,float>
                    ((float*)derData.getMemory(), (float const*)data.getMemory(), (float const*)derPooled.getMemory(),
                     derData.getHeight(), derData.getWidth(), derData.getDepth(), derData.getSize(),
                     numTotBins,
                     levels.getNumElements(), (const float *)levels.getMemory(),
                     ROIs.getNumElements()/5, (const float *)ROIs.getMemory()) ;
        }
        if (status == vlErrorCuda) {
            context.setError(context.getCudaHelper().catchCudaError("spp_*_backward: ")) ;
        }
        break ;
#endif
    }
    return context.passError(status, "nnspp_backward: ") ;
}
