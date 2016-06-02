// @file spp_cpu.cpp
// @brief SPP block implementation (GPU)
// @author Hakan Bilen 

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "nnspp.hpp"

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
    {"NumBins",          1,   opt_numbins },
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
    int numLevels = 0;
    int numROIs = 0;
    int numTotBins = 0;
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
        case opt_numbins :
            if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
                mexErrMsgTxt("NUMBINS is not a plain matrix.") ;
            }
            numTotBins = (int)mxGetPr(optarg)[0] ;
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
    vl::MexTensor dlevels(context) ;

    // load pyramid levels and rois
    dlevels.init(in[IN_LEVELS]);
    dROIs.init(in[IN_ROIS]);


    if (verbosity > 0) {
        mexPrintf("vl_nnspp.cu: numTotBins %d\n",numTotBins);
        mexPrintf("levels %d %d %d %d\n",dlevels.getWidth(),dlevels.getHeight(),dlevels.getDepth(),dlevels.getSize());
        mexPrintf("dROIs %d %d %d %d\n",dROIs.getWidth(),dROIs.getHeight(),dROIs.getDepth(),dROIs.getSize());
        mexPrintf("vl_nnspp.cu: numTotBins %d\n",numTotBins);
    }
    //  mexErrMsgTxt("levels and dROIs are not in right size.") ;

    data.init(in[IN_DATA]) ;
    if (backMode) { derOutput.init(in[IN_DEROUTPUT]) ; }

    if (backMode && ! vl::areCompatible(data, derOutput)) {
        mexErrMsgTxt("DATA and DEROUTPUT are not both CPU or GPU arrays.") ;
    }

    numLevels = dlevels.getNumElements();
    if (numLevels<=0) {
        mexErrMsgTxt("LEVELS has zero elements.") ;
    }

    numROIs = dROIs.getNumElements();


    if (numROIs % 5 != 0) {
        mexErrMsgTxt("Wrong number of elements in ROIS.") ;
    }

    if (numROIs<=0) {
        mexErrMsgTxt("ROIs has zero elements.") ;
    }

    numROIs /= 5;


    if(numTotBins<=0) {
        mexPrintf("numTotBins %d\n",numTotBins);
        mexErrMsgTxt("numTotBins is wrong.") ;
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
                                  dlevels,
                                  dROIs) ;

    } else {
        error = vl::nnspp_backward(context,
                                   derData, data, derOutput,
                                   method,
                                   numTotBins,
                                   dlevels,
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

    //  free(levels);
}
