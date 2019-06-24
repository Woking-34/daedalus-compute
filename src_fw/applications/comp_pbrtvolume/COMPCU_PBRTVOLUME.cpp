#include "COMPCU_RTOW.h"
#include "COMP_RTOW_TYPES.h"

#include "system/log.h"
#include "system/filesystem.h"

extern "C" void cuK_rtow
(
	int max_x, int max_y, int sampleNum, int sphereNum, void* cuMSpheres, void* cuMMaterials, void* cuMCamera, void* cuMSeed0, void* cuMSeed1, cudaSurfaceObject_t viewCudaSurfaceObject,
	unsigned int gSizeX, unsigned int gSizeY, unsigned int lSizeX, unsigned int lSizeY
);

COMPCU_RTOW::COMPCU_RTOW()
{
	cuMCamera = NULL;
	cuMSeed0 = NULL;
	cuMSeed1 = NULL;
	cuMSpheres = NULL;
	cuMMaterials = NULL;
}

COMPCU_RTOW::~COMPCU_RTOW()
{

}

void COMPCU_RTOW::init()
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

	CHECK_CU(cudaMalloc((void**)&cuMCamera, 8*4 * sizeof(float)));
	CHECK_CU(cudaMemcpy(cuMCamera, cameraArray, 8*4 * sizeof(float), cudaMemcpyHostToDevice));

	CHECK_CU(cudaMalloc((void**)&cuMSpheres, sphereNum * sizeof(rtow_sphere)));
	CHECK_CU(cudaMemcpy(cuMSpheres, sphereArrayHost, sphereNum * sizeof(rtow_sphere), cudaMemcpyHostToDevice));

	CHECK_CU(cudaMalloc((void**)&cuMMaterials, materialNum * sizeof(rtow_material)));
	CHECK_CU(cudaMemcpy(cuMMaterials, materialArrayHost, materialNum * sizeof(rtow_material), cudaMemcpyHostToDevice));

	CHECK_CU(cudaMalloc((void**)&cuMSeed0, launchW*launchH * sizeof(unsigned int)));
	CHECK_CU(cudaMemcpy(cuMSeed0, seed0, launchW*launchH * sizeof(unsigned int), cudaMemcpyHostToDevice));

	CHECK_CU(cudaMalloc((void**)&cuMSeed1, launchW*launchH * sizeof(unsigned int)));
	CHECK_CU(cudaMemcpy(cuMSeed1, seed1, launchW*launchH * sizeof(unsigned int), cudaMemcpyHostToDevice));

	if (useInterop)
	{
		CHECK_CU(cudaGraphicsGLRegisterImage(&viewCudaResource, interopId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
		CHECK_CU(cudaGraphicsMapResources(1, &viewCudaResource));

		CHECK_CU(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0, 0));

		cudaResourceDesc viewCudaArrayResourceDesc;
		memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
		viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
		viewCudaArrayResourceDesc.res.array.array = viewCudaArray;

		CHECK_CU(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));

		CHECK_CU(cudaGraphicsUnmapResources(1, &viewCudaResource));
	}
	else
	{
		// TODO: fix this
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		//CHECK_CU(cudaMallocArray(&viewCudaArray, &channelDesc, launchW, launchH), cudaArraySurfaceLoadStore);
		CHECK_CU(cudaMalloc3DArray(&viewCudaArray, &channelDesc, make_cudaExtent(launchW, launchH, 0), cudaArraySurfaceLoadStore));

		cudaResourceDesc viewCudaArrayResourceDesc;
		memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
		viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
		viewCudaArrayResourceDesc.res.array.array = viewCudaArray;

		CHECK_CU(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));
	}

	CHECK_CU(cudaDeviceSynchronize());
}

void COMPCU_RTOW::terminate()
{
	if (useInterop)
	{
		CHECK_CU(cudaDestroySurfaceObject(viewCudaSurfaceObject));
		CHECK_CU(cudaGraphicsUnregisterResource(viewCudaResource));
	}
	else
	{
		CHECK_CU(cudaFreeArray(viewCudaArray));
		CHECK_CU(cudaDestroySurfaceObject(viewCudaSurfaceObject));
	}

	if (cuMSpheres)
	{
		CHECK_CU(cudaFree(cuMSpheres));
		cuMSpheres = NULL;
	}
	if (cuMMaterials)
	{
		CHECK_CU(cudaFree(cuMMaterials));
		cuMMaterials = NULL;
	}

	if (cuMCamera)
	{
		CHECK_CU(cudaFree(cuMCamera));
		cuMCamera = NULL;
	}

	if(cuMSeed0)
	{
		CHECK_CU(cudaFree(cuMSeed0));
		cuMSeed0 = NULL;
	}
	if(cuMSeed1)
	{
		CHECK_CU(cudaFree(cuMSeed1));
		cuMSeed1 = NULL;
	}

	CHECK_CU(cudaDeviceReset());
}

void COMPCU_RTOW::compute()
{
	if(useInterop)
	{
		CHECK_CU(cudaGraphicsMapResources(1, &viewCudaResource));
	}

	if(isDynamicCamera)
	{
		CHECK_CU(cudaMemcpy(cuMCamera, cameraArray, 8 * 4 * sizeof(float), cudaMemcpyHostToDevice));
	}

	{
		cuK_rtow( launchW, launchH, sampleNum, sphereNum, cuMSpheres, cuMMaterials, cuMCamera, cuMSeed0, cuMSeed1, viewCudaSurfaceObject, launchW, launchH, wgsX, wgsY );
	}

	if(useInterop)
	{
		CHECK_CU(cudaGraphicsUnmapResources(1, &viewCudaResource));
	}

	CHECK_CU(cudaDeviceSynchronize());
}

void COMPCU_RTOW::download()
{
	// TODO: fix this
	//CHECK_CU(cudaMemcpy(outputFLT, cuMImage, launchW*launchH*sizeof(float4), cudaMemcpyDeviceToHost));
	//CHECK_CU(cudaMemcpy2DFromArray(outputFLT, launchW * sizeof(float4), viewCudaArray, 0, 0, launchW * sizeof(float4), launchH, cudaMemcpyDeviceToHost));
	//CHECK_CU(cudaMemcpyFromArray(outputFLT, viewCudaArray, 0, 0, launchW * launchH * sizeof(float4), cudaMemcpyDeviceToHost));

	//cudaMemcpy3DParms myparms = { 0 };
	//myparms.srcArray = viewCudaArray;
	//myparms.srcPos = make_cudaPos(0, 0, 0);
	//myparms.dstPtr = make_cudaPitchedPtr(outputFLT, launchW * sizeof(float4), launchW, launchH);
	//myparms.dstPos = make_cudaPos(0, 0, 0);
	//myparms.extent = make_cudaExtent(launchW, launchH, 0);
	//myparms.kind = cudaMemcpyDeviceToHost;
	//CHECK_CU(cudaMemcpy3D(&myparms));
}