// @file vl_nnspp.cu
// @brief Spatial Pyramid Pooling MEX wrapper
// @author Hakan Bilen 
// @author Andrea Vedaldi
/*
Copyright (C) 2016- Hakan Bilen and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/** this is the mex-wrapper -- entry-point from matlab to cuda */

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnspp.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
    opt_numbins=0,
    opt_method,
    opt_verbose,
} ;

/* options */
vlmxOption  options [] = {
    {"Method",           1,   opt_method },
    {"Verbose",          0,   opt_verbose },
    {0,                  0,   0           }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
    context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
    IN_DATA = 0, IN_LEVELS, IN_ROIS, IN_DEROUTPUT, IN_END
} ;

enum {
    OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
    size_t numLevels = 0;
    size_t numROIs = 0;
    size_t numTotBins = 0;
    // 0 -> max, 1 -> avg
    int method = 0;

    bool backMode = false ;

    int verbosity = 0 ;
    int opt ;
    int next = IN_END ;
    mxArray const *optarg ;

    /* -------------------------------------------------------------- */
    /*                                            Check the arguments */
    /* -------------------------------------------------------------- */

    mexAtExit(atExit) ;

    if (nin < 3) {
        mexErrMsgTxt("The arguments are less than three.") ;
    }

    if (nin > 3 && vlmxIsString(in[3],-1)) {
        next = 3 ;
        backMode = 0 ;
    } else {
        backMode = (nin >= 4) ;
    }

    while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
        switch (opt) {
        case opt_verbose :
            ++ verbosity ;
            break ;
        case opt_method :
            if (!vlmxIsString(optarg,-1)) {
                vlmxError(vlmxErrInvalidArgument, "METHOD is not a string.") ;
            }
            if (vlmxIsEqualToStringI(optarg, "max")) {
                method = 0 ;
            } else if (vlmxIsEqualToStringI(optarg, "avg")) {
                method = 1 ;
            } else {
                vlmxError(vlmxErrInvalidArgument, "METHOD is not a supported method.") ;
            }
        default:
            break ;
        }
    }


    vl::MexTensor data(context) ;
    vl::MexTensor derOutput(context) ;

    vl::MexTensor dROIs(context) ;
    vl::MexTensor pyrLevels(context) ;

    // load pyramid levels and rois
    pyrLevels.init(in[IN_LEVELS]);
    dROIs.init(in[IN_ROIS]);

    size_t elemPL = mxGetNumberOfElements(in[IN_LEVELS]);

    for (size_t i=0;i<elemPL;i++)
      numTotBins += mxGetPr(in[IN_LEVELS])[i];

    if(numTotBins<=0) {
        mexPrintf("numTotBins %d\n",numTotBins);
        mexErrMsgTxt("numTotBins is wrong.") ;
    }


    if (verbosity > 0) {
        mexPrintf("vl_nnspp.cu: numTotBins %d\n",numTotBins);
        mexPrintf("levels %d %d %d %d\n",pyrLevels.getWidth(),pyrLevels.getHeight(),pyrLevels.getDepth(),pyrLevels.getSize());
        mexPrintf("dROIs %d %d %d %d\n",dROIs.getWidth(),dROIs.getHeight(),dROIs.getDepth(),dROIs.getSize());
        mexPrintf("vl_nnspp.cu: numTotBins %d\n",numTotBins);
    }

    data.init(in[IN_DATA]) ;
    if (backMode) { derOutput.init(in[IN_DEROUTPUT]) ; }

    if (backMode && ! vl::areCompatible(data, derOutput)) {
        mexErrMsgTxt("DATA and DEROUTPUT are not both CPU or GPU arrays.") ;
    }

    numLevels = pyrLevels.getNumElements();
    if (numLevels<=0) {
        mexErrMsgTxt("LEVELS has zero elements.") ;
    }

    numROIs = dROIs.getWidth();


    if (dROIs.getWidth() != 5) {
        mexErrMsgTxt("Wrong number of elements in ROIS.") ;
    }

    if (numROIs<=0) {
        mexErrMsgTxt("ROIs has zero elements.") ;
    }


    if (verbosity > 0) {
        mexPrintf("numTotBins %d depth %d numROIs %d\n",numTotBins,data.getDepth(),numROIs);
    }
    /* Get the output geometry */
    vl::TensorShape outputShape(1, numTotBins,
                                data.getDepth(),
                                numROIs) ;

    vl::TensorShape dataShape = data.getShape();

    if(dataShape.getNumDimensions()<4) {
        dataShape.reshape(4);
    }

    /* Create output buffers */
    vl::Device deviceType = data.getDeviceType() ;
    vl::Type dataType = data.getDataType() ;
    vl::MexTensor output(context) ;
    vl::MexTensor derData(context) ;

    if (verbosity > 0) {
        vl::print("vl_nnspp: data: ", data) ;
        if (backMode) {
            vl::print("vl_nnspp: derOutput: ", derOutput) ;
            vl::print("vl_nnspp: derData: ", derData) ;
        } else {
            vl::print("vl_nnspp: output: ", output) ;
        }
    }



    if (!backMode) {
        output.initWithZeros(deviceType, dataType, outputShape) ;
    } else {
        derData.initWithZeros(deviceType, dataType, dataShape) ;
    }

    if (verbosity > 0) {
        mexPrintf("vl_spp: %s; %s", backMode?"backward":"forward", (data.getDeviceType()==vl::GPU) ? "GPU" : "CPU") ;
        mexPrintf("\nvl_spp: method %d numLevels %d; numROIs %d numTotBins %d\n", method, numLevels, numROIs, numTotBins);
    }

    /* -------------------------------------------------------------- */
    /*                                                    Do the work */
    /* -------------------------------------------------------------- */

    vl::Error error ;
    if (!backMode) {
        error = vl::nnspp_forward(context,
                                  output, data,
                                  method,
                                  numTotBins,
                                  pyrLevels,
                                  dROIs) ;

    } else {
        error = vl::nnspp_backward(context,
                                   derData, data, derOutput,
                                   method,
                                   numTotBins,
                                   pyrLevels,
                                   dROIs) ;
    }

    /* -------------------------------------------------------------- */
    /*                                                         Finish */
    /* -------------------------------------------------------------- */

    if (error != vl::vlSuccess) {
        mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
    }
    if (backMode) {
        out[OUT_RESULT] = derData.relinquish() ;
    } else {
        out[OUT_RESULT] = output.relinquish() ;
    }
}
