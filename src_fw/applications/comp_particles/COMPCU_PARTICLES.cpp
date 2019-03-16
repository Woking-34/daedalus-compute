#include "COMPCU_PARTICLES.h"

#include "system/log.h"
#include "system/filesystem.h"

extern "C" void cuK_reset
(
	void* hList, int gSizeX, int lSizeX
);

extern "C" void cuK_create
(
	void* pos, void* hList, void* pList, float particleRadius, int gridSize, int gSizeX, int lSizeX
);

extern "C" void cuK_collide
(
	void* hList, void* pList, void* pos, void* vel0, void* vel1, float particleRadius, int gridSize, int gSizeX, int lSizeX
);

extern "C" void cuK_integrate
(
	void* pos, void* vel0, void* vel1, float particleRadius, int gSizeX, int lSizeX
);

COMPCU_PARTICLES::COMPCU_PARTICLES()
{
	cuMPos = NULL;

	cuMVel0 = NULL;
	cuMVel1 = NULL;
	cuMHList = NULL;
	cuMPList = NULL;
}

COMPCU_PARTICLES::~COMPCU_PARTICLES()
{

}

void COMPCU_PARTICLES::init()
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

	CHECK_CU(cudaMalloc((void**)&cuMPos, numParticles*sizeof(float4)));

	CHECK_CU(cudaMalloc((void**)&cuMVel0, numParticles*sizeof(float4)));
	CHECK_CU(cudaMalloc((void**)&cuMVel1, numParticles*sizeof(float4)));

	CHECK_CU(cudaMalloc((void**)&cuMHList, numGridCells*sizeof(int)));
	CHECK_CU(cudaMalloc((void**)&cuMPList, numParticles*sizeof(int)));

	CHECK_CU(cudaMemcpy(cuMPos, bufferPos, numParticles*sizeof(float4), cudaMemcpyHostToDevice));
	CHECK_CU(cudaMemcpy(cuMVel0, bufferVel, numParticles*sizeof(float4), cudaMemcpyHostToDevice));
	CHECK_CU(cudaMemcpy(cuMVel1, bufferVel, numParticles*sizeof(float4), cudaMemcpyHostToDevice));

	if (useInterop)
	{
		CHECK_CU(cudaGraphicsGLRegisterBuffer(&cuVBOPos, vboPos, cudaGraphicsMapFlagsWriteDiscard));
	}

	CHECK_CU(cudaDeviceSynchronize());
}

void COMPCU_PARTICLES::terminate()
{
	if (useInterop)
	{
		CHECK_CU(cudaGraphicsUnregisterResource(cuVBOPos));
	}

	if (cuMPos)
	{
		CHECK_CU(cudaFree(cuMPos));
		cuMPos = NULL;
	}
	if (cuMVel0)
	{
		CHECK_CU(cudaFree(cuMVel0));
		cuMVel0 = NULL;
	}
	if (cuMVel1)
	{
		CHECK_CU(cudaFree(cuMVel1));
		cuMVel1 = NULL;
	}

	if (cuMHList)
	{
		CHECK_CU(cudaFree(cuMHList));
		cuMHList = NULL;
	}
	if (cuMPList)
	{
		CHECK_CU(cudaFree(cuMPList));
		cuMPList = NULL;
	}
	
	CHECK_CU(cudaDeviceReset());
}

void COMPCU_PARTICLES::compute()
{
	void* cuglPos = nullptr;

	if (useInterop)
	{
		cudaError cudaStatus = cudaSuccess;
	
		cudaStatus = cudaGraphicsMapResources(1, &cuVBOPos, 0);
		CHECK_CU(cudaStatus);

		size_t size = numParticles * 4 * sizeof(float);

		cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&cuglPos, &size, cuVBOPos);
		CHECK_CU(cudaStatus);
	}

	void* posPtr = NULL;

	if (useInterop)
	{
		posPtr = cuglPos;
	}
	else
	{
		posPtr = cuMPos;
	}

	for (int i = 0; i < 5; ++i)
	{
		cuK_reset(cuMHList, numGridCellsPaddded, wgsX);
		cuK_create(posPtr, cuMHList, cuMPList, particleRadius, gridCells, numParticles, wgsX);
		cuK_collide(cuMHList, cuMPList, posPtr, cuMVel0, cuMVel1, particleRadius, gridCells, numParticles, wgsX);
		cuK_integrate(posPtr, cuMVel0, cuMVel1, particleRadius, numParticles, wgsX);
	}

	CHECK_CU(cudaDeviceSynchronize());

	if (useInterop)
	{
		cudaGraphicsUnmapResources(1, &cuVBOPos, 0);
	}
}

void COMPCU_PARTICLES::download()
{
	CUresult cuRes = CUDA_SUCCESS;

	CHECK_CU(cudaMemcpy(bufferPos, cuMPos, numParticles*sizeof(float4), cudaMemcpyDeviceToHost));
	CHECK_CU(cudaDeviceSynchronize());
}