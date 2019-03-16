#include "COMPCU_DEFORM.h"

#include "system/log.h"
#include "system/filesystem.h"

extern "C" void cuK_deform
(
	void* g_pos, const int currDeform, const float currTime,
	const float gridWidth, const float gridWidthDt, const float gridLength, const float gridLengthDt,
	int gSizeX, int gSizeY, int lSizeX, int lSizeY
);

COMPCU_DEFORM::COMPCU_DEFORM()
{
	cuMVerts = NULL;
	cuVBOVerts = NULL;
}

COMPCU_DEFORM::~COMPCU_DEFORM()
{

}

void COMPCU_DEFORM::init()
{
	cudaError_t cuStatus = cudaSuccess;

	int deviceCount;
	cuStatus = cudaGetDeviceCount(&deviceCount);
	CHECK_CU(cuStatus);

	if (deviceCount != 0)
	{
		std::cout << "Found " << deviceCount << " CUDA Capable Device(s)." << std::endl;
		std::cout << std::endl;
	}

	for (int deviceCurr = 0; deviceCurr < deviceCount; ++deviceCurr)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, deviceCurr);

		std::cout << "Device " << deviceCurr << ": " << deviceProp.name << std::endl;

		int runtimeVersion;
		cudaRuntimeGetVersion(&runtimeVersion);
		std::cout << "CUDA Runtime Version:\t\t" << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;
		std::cout << "CUDA Compute Capability:\t" << deviceProp.major << "." << deviceProp.minor << std::endl;

		std::cout << std::endl;
	}

	CHECK_CU(cudaSetDevice(0));

	CHECK_CU(cudaMalloc((void**)&cuMVerts, 2*launchW*launchH*sizeof(float4)));
	
	if (useInterop)
	{
		CHECK_CU(cudaGraphicsGLRegisterBuffer(&cuVBOVerts, vbo, cudaGraphicsMapFlagsWriteDiscard));
	}

	CHECK_CU(cudaDeviceSynchronize());
}

void COMPCU_DEFORM::terminate()
{
	if (useInterop)
	{
		CHECK_CU(cudaGraphicsUnregisterResource(cuVBOVerts));
	}

	if (cuMVerts)
	{
		CHECK_CU(cudaFree(cuMVerts));
		cuMVerts = NULL;
	}
	
	CHECK_CU(cudaDeviceReset());
}

void COMPCU_DEFORM::compute(int currDeform, float currTime)
{
	void* cuglMemVerts = nullptr;
	
	if (useInterop)
	{
		cudaError cudaStatus = cudaSuccess;
	
		cudaStatus = cudaGraphicsMapResources(1, &cuVBOVerts, 0);
		CHECK_CU(cudaStatus);
		
		size_t size = launchW*launchH * 8 * sizeof(float);

		cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&cuglMemVerts, &size, cuVBOVerts);
		CHECK_CU(cudaStatus);
	}

	if (useInterop)
	{
		cuK_deform(cuglMemVerts, currDeform, currTime, sizeX, stepX, sizeY, stepY, launchW, launchH, wgsX, wgsY);
	}
	else
	{
		cuK_deform(cuMVerts, currDeform, currTime, sizeX, stepX, sizeY, stepY, launchW, launchH, wgsX, wgsY);
	}

	CHECK_CU(cudaDeviceSynchronize());

	if (useInterop)
	{
		cudaGraphicsUnmapResources(1, &cuVBOVerts, 0);
	}
}

void COMPCU_DEFORM::download()
{
	CUresult cuRes = CUDA_SUCCESS;

	CHECK_CU(cudaMemcpy(bufferVertices, cuMVerts, 2*launchW*launchH*sizeof(float4), cudaMemcpyDeviceToHost));
	CHECK_CU(cudaDeviceSynchronize());
}