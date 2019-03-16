#include "COMPCU_CLOTH.h"

#include "system/log.h"
#include "system/filesystem.h"

extern "C" void cuK_cloth
(
	void* g_pos_in, void* g_pos_old_in, void* g_pos_out, void* g_pos_old_out, void* g_normals_out,
	void* mass, float damp, float dt, float stepX, float stepY,
	int gSizeX, int gSizeY, int lSizeX, int lSizeY
);

COMPCU_CLOTH::COMPCU_CLOTH()
{
	cuMMass = NULL;

	cuMIn = NULL;
	cuMInOld = NULL;
	cuMOut = NULL;
	cuMOutOld = NULL;
	cuMNormals = NULL;
}

COMPCU_CLOTH::~COMPCU_CLOTH()
{

}

void COMPCU_CLOTH::init()
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

	CHECK_CU(cudaMalloc((void**)&cuMIn, launchW*launchH*sizeof(float4)));
	CHECK_CU(cudaMalloc((void**)&cuMInOld, launchW*launchH*sizeof(float4)));

	CHECK_CU(cudaMalloc((void**)&cuMOut, launchW*launchH*sizeof(float4)));
	CHECK_CU(cudaMalloc((void**)&cuMOutOld, launchW*launchH*sizeof(float4)));

	CHECK_CU(cudaMemcpy(cuMIn, bufferPositions, launchW*launchH*sizeof(float4), cudaMemcpyHostToDevice));
	CHECK_CU(cudaMemcpy(cuMInOld, bufferPositions, launchW*launchH*sizeof(float4), cudaMemcpyHostToDevice));

	CHECK_CU(cudaMalloc((void**)&cuMNormals, launchW*launchH*sizeof(float4)));
	CHECK_CU(cudaMemcpy(cuMNormals, bufferNormals, launchW*launchH*sizeof(float4), cudaMemcpyHostToDevice));

	CHECK_CU(cudaMalloc((void**)&cuMMass, launchW*launchH*sizeof(float)));
	CHECK_CU(cudaMemcpy(cuMMass, bufferWeights, launchW*launchH*sizeof(float), cudaMemcpyHostToDevice));

	if (useInterop)
	{
		CHECK_CU(cudaGraphicsGLRegisterBuffer(&cuVBOIn, vboInCurr, cudaGraphicsMapFlagsWriteDiscard));
		CHECK_CU(cudaGraphicsGLRegisterBuffer(&cuVBOInOld, vboInPrev, cudaGraphicsMapFlagsWriteDiscard));
		CHECK_CU(cudaGraphicsGLRegisterBuffer(&cuVBOOut, vboOutCurr, cudaGraphicsMapFlagsWriteDiscard));
		CHECK_CU(cudaGraphicsGLRegisterBuffer(&cuVBOOutOld, vboOutPrev, cudaGraphicsMapFlagsWriteDiscard));

		CHECK_CU(cudaGraphicsGLRegisterBuffer(&cuVBONormals, vboNormals, cudaGraphicsMapFlagsWriteDiscard));
	}

	CHECK_CU(cudaDeviceSynchronize());
}

void COMPCU_CLOTH::terminate()
{
	if (useInterop)
	{
		CHECK_CU(cudaGraphicsUnregisterResource(cuVBOIn));
		CHECK_CU(cudaGraphicsUnregisterResource(cuVBOInOld));
		CHECK_CU(cudaGraphicsUnregisterResource(cuVBOOut));
		CHECK_CU(cudaGraphicsUnregisterResource(cuVBOOutOld));

		CHECK_CU(cudaGraphicsUnregisterResource(cuVBONormals));
	}

	if (cuMIn)
	{
		CHECK_CU(cudaFree(cuMIn));
		cuMIn = NULL;
	}
	if (cuMInOld)
	{
		CHECK_CU(cudaFree(cuMInOld));
		cuMInOld = NULL;
	}
	if (cuMOut)
	{
		CHECK_CU(cudaFree(cuMOut));
		cuMOut = NULL;
	}
	if (cuMOutOld)
	{
		CHECK_CU(cudaFree(cuMOutOld));
		cuMOutOld = NULL;
	}

	if (cuMOutOld)
	{
		CHECK_CU(cudaFree(cuMNormals));
		cuMNormals = NULL;
	}
	
	if (cuMMass)
	{
		CHECK_CU(cudaFree(cuMMass));
		cuMMass = NULL;
	}

	CHECK_CU(cudaDeviceReset());
}

void COMPCU_CLOTH::compute()
{
	void* cuglMemIn = nullptr;
	void* cuglMemInOld = nullptr;
	void* cuglMemOut = nullptr;
	void* cuglMemOutOld = nullptr;
	void* cuglMemNormals = nullptr;

	if (useInterop)
	{
		cudaError cudaStatus = cudaSuccess;
	
		cudaStatus = cudaGraphicsMapResources(1, &cuVBOIn, 0);
		CHECK_CU(cudaStatus);
		cudaStatus = cudaGraphicsMapResources(1, &cuVBOInOld, 0);
		CHECK_CU(cudaStatus);

		cudaStatus = cudaGraphicsMapResources(1, &cuVBOOut, 0);
		CHECK_CU(cudaStatus);
		cudaStatus = cudaGraphicsMapResources(1, &cuVBOOutOld, 0);
		CHECK_CU(cudaStatus);

		cudaStatus = cudaGraphicsMapResources(1, &cuVBONormals, 0);
		CHECK_CU(cudaStatus);

		size_t size = launchW*launchH * 4 * sizeof(float);

		cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&cuglMemIn, &size, cuVBOIn);
		CHECK_CU(cudaStatus);
		cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&cuglMemInOld, &size, cuVBOInOld);
		CHECK_CU(cudaStatus);

		cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&cuglMemOut, &size, cuVBOOut);
		CHECK_CU(cudaStatus);
		cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&cuglMemOutOld, &size, cuVBOOutOld);
		CHECK_CU(cudaStatus);

		cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&cuglMemNormals, &size, cuVBONormals);
		CHECK_CU(cudaStatus);
	}

	if (useInterop)
	{
		cuK_cloth(cuglMemIn, cuglMemInOld, cuglMemOut, cuglMemOutOld, cuglMemNormals, cuMMass, damp, dt, stepX, stepY, launchW, launchH, wgsX, wgsY);
	}
	else
	{
		cuK_cloth(cuMIn, cuMInOld, cuMOut, cuMOutOld, cuMNormals, cuMMass, damp, dt, stepX, stepX, launchW, launchH, wgsX, wgsY);
	}

	CHECK_CU(cudaDeviceSynchronize());

	if (useInterop)
	{
		cudaGraphicsUnmapResources(1, &cuVBOIn, 0);
		cudaGraphicsUnmapResources(1, &cuVBOInOld, 0);

		cudaGraphicsUnmapResources(1, &cuVBOOut, 0);
		cudaGraphicsUnmapResources(1, &cuVBOOutOld, 0);

		cudaGraphicsUnmapResources(1, &cuVBONormals, 0);
	}

	std::swap(cuMIn, cuMOut);
	std::swap(cuMInOld, cuMOutOld);

	std::swap(cuVBOIn, cuVBOOut);
	std::swap(cuVBOInOld, cuVBOOutOld);
	
	std::swap(vboInCurr, vboOutCurr);
	std::swap(vboInPrev, vboOutPrev);
}

void COMPCU_CLOTH::download()
{
	CUresult cuRes = CUDA_SUCCESS;

	CHECK_CU(cudaMemcpy(bufferPositions, cuMIn, launchW*launchH*sizeof(float4), cudaMemcpyDeviceToHost));
	CHECK_CU(cudaDeviceSynchronize());

	CHECK_CU(cudaMemcpy(bufferNormals, cuMNormals, launchW*launchH*sizeof(float4), cudaMemcpyDeviceToHost));
	CHECK_CU(cudaDeviceSynchronize());
}