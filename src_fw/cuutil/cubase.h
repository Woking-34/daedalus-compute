#ifndef CUBASE_H
#define CUBASE_H

//#ifdef CUDA_FOUND

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "system/log.h"

#define CHECK_CU(cuStatus) if(cuStatus != cudaSuccess) { LOG_ERR( LogLine() << "File: " << __FILE__ << "\n" << "Line: " << __LINE__ << "\n" << "CUDA error: " << cudaGetErrorString( cuStatus ) << "\n"); }

//#endif

#endif