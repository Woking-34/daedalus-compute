#include "COMPCL_PBRTVOLUME.h"

#include "system/log.h"
#include "system/filesystem.h"

COMPCL_PBRTVOLUME::COMPCL_PBRTVOLUME()
{
	selectedPlatformIndex = -1;
	selectedDeviceIndex = -1;

	clContext = 0;
	clQueue = 0;
	
	clProg_pbrtvolume = 0;
	clK_render = 0;

	clMImage = 0;
	clMCamera = 0;
	clMVolumeData = 0;
	clMRaster2Camera = 0;
	clMCamera2World = 0;
}

COMPCL_PBRTVOLUME::~COMPCL_PBRTVOLUME()
{

}

void COMPCL_PBRTVOLUME::init()
{
	int clewOK = initClew();
	if( clewOK != 0 )
	{
		std::cout << "initClew() failed!" << std::endl;
		exit(-1);
	}

	cl.init();

	selectedPlatformIndex = useCLPId;
	selectedDeviceIndex = useCLDId;

	if(useCLPId != -1 && useCLDId != -1)
	{
		cl.selectePlatformDevice(selectedPlatformIndex, selectedDeviceIndex, selectedPlatformID, selectedDeviceID);
	}
	else if(useCLGDevice)
	{
		cl.selectePlatformInteropDevice(selectedPlatformIndex, selectedDeviceIndex, selectedPlatformID, selectedDeviceID);
	}
	else
	{
		LOG_ERR( "Invalid CL platform/device configuration!" );
		assert(0);
	}

	bool hasGLSharing = cl.platforms[selectedPlatformIndex].devices[selectedDeviceIndex].isExtSupported("cl_khr_gl_sharing");
	
	// force cl cpu to disable sharing
	if(cl.platforms[selectedPlatformIndex].devices[selectedDeviceIndex].isCPU())
	{
		useInterop = false;
	}

	useInterop = useInterop && hasGLSharing;

	std::vector<cl_context_properties> clContextProps = createContextProps(selectedPlatformID, useInterop);

	cl_int clStatus = 0;

	cl_int deviceCount = 1;
	clContext = clCreateContext(&clContextProps[0], deviceCount, &selectedDeviceID, NULL, NULL, &clStatus);
	CHECK_CL(clStatus);

	clQueue = clCreateCommandQueue(clContext, selectedDeviceID, 0, &clStatus);
	CHECK_CL(clStatus);
	
	std::string filename = "pbrtvolume/pbrtvolume.cl";

	std::string clFileName = FileSystem::GetKernelsCLFolder() + filename;
	std::string clProgStr = loadStr(clFileName);

	const char* clProgramStr = clProgStr.c_str();
	size_t clProgramSize = clProgStr.size();

	clProg_pbrtvolume = clCreateProgramWithSource(clContext, 1, &clProgramStr, &clProgramSize, &clStatus);
	CHECK_CL(clStatus);

	//clStatus = clBuildProgram(clProg_root, 0, NULL, NULL, NULL, NULL);
	clStatus = clBuildProgram(clProg_pbrtvolume, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
	if(clStatus == CL_BUILD_PROGRAM_FAILURE)
	{
		size_t buildLogSize = 0;
		clStatus =  clGetProgramBuildInfo(clProg_pbrtvolume, selectedDeviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
		CHECK_CL(clStatus);

		char* buildLog = new char[buildLogSize];
		clStatus = clGetProgramBuildInfo(clProg_pbrtvolume, selectedDeviceID, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
		CHECK_CL(clStatus);

		std::cout << buildLog << std::endl;
	}

	clK_render = clCreateKernel(clProg_pbrtvolume, "render", &clStatus);
	CHECK_CL(clStatus);
		
	if(useInterop)
	{
		clMImage = clCreateFromGLTexture2D(clContext, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, interopId, &clStatus);
		CHECK_CL(clStatus);
	}
	else
	{
		const cl_image_format imgFormatFLT = {CL_RGBA, CL_FLOAT};
		const size_t rowPitch = 0;

		clMImage = clCreateImage2D(clContext, CL_MEM_READ_WRITE, &imgFormatFLT, launchW, launchH, rowPitch, NULL, &clStatus);
		CHECK_CL(clStatus);
	}

	clMCamera = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 8*4 * sizeof(float), cameraArray, &clStatus);
	CHECK_CL(clStatus);
	clMVolumeData = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vX*vY*vZ * sizeof(float), volumeData, &clStatus);
	CHECK_CL(clStatus);
	clMRaster2Camera = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), raster2camera, &clStatus);
	CHECK_CL(clStatus);
	clMCamera2World = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), camera2world, &clStatus);
	CHECK_CL(clStatus);

	clStatus = clFinish( clQueue );
	CHECK_CL(clStatus);
}

void COMPCL_PBRTVOLUME::terminate()
{
	if(clMImage)
	{
		CHECK_CL( clReleaseMemObject(clMImage) );
		clMImage = 0;
	}

	if(clMCamera)
	{
		CHECK_CL( clReleaseMemObject(clMCamera) );
		clMCamera = 0;
	}

	if(clMVolumeData)
	{
		CHECK_CL( clReleaseMemObject(clMVolumeData) );
		clMVolumeData = 0;
	}

	if(clMRaster2Camera)
	{
		CHECK_CL( clReleaseMemObject(clMRaster2Camera) );
		clMRaster2Camera = 0;
	}

	if (clMCamera2World)
	{
		CHECK_CL(clReleaseMemObject(clMCamera2World));
		clMCamera2World = 0;
	}
	
	if(clK_render)
	{
		CHECK_CL( clReleaseKernel(clK_render) );
		clK_render = 0;
	}

	if(clProg_pbrtvolume)
	{
		CHECK_CL(  clReleaseProgram(clProg_pbrtvolume) );
		clProg_pbrtvolume = 0;
	}

	if(clQueue)
	{
		CHECK_CL( clReleaseCommandQueue(clQueue) );
		clQueue = 0;
	}

	if(clContext)
	{
		CHECK_CL( clReleaseContext(clContext) );
		clContext = 0;
	}
}

void COMPCL_PBRTVOLUME::compute()
{
	cl_int clStatus = CL_SUCCESS;

	if(useInterop)
	{
		clStatus = clEnqueueAcquireGLObjects(clQueue, 1, &clMImage, 0, NULL, NULL );
		CHECK_CL(clStatus);
	}

	if(isDynamicCamera)
	{
		clStatus = clEnqueueWriteBuffer(clQueue, clMCamera, CL_TRUE, 0, 8 * 4 * sizeof(float), cameraArray, 0, NULL, NULL);
		CHECK_CL(clStatus);
	}

	{
		size_t globalWS[3] = { launchW, launchH, 1 };
		//size_t localWS[3] = { wgsX, wgsY, 1 };
		size_t* localWS = NULL;

		cl_int3 nVoxels;
		nVoxels.s[0] = vX;
		nVoxels.s[1] = vY;
		nVoxels.s[2] = vZ;
		
		clStatus |= clSetKernelArg(clK_render, 0, sizeof(cl_int),  (void*)&launchW);
		clStatus |= clSetKernelArg(clK_render, 1, sizeof(cl_int),  (void*)&launchH);
		clStatus |= clSetKernelArg(clK_render, 2, sizeof(cl_int3), (void*)&nVoxels);
		clStatus |= clSetKernelArg(clK_render, 3, sizeof(cl_mem),  (void*)&clMVolumeData);
		clStatus |= clSetKernelArg(clK_render, 4, sizeof(cl_mem),  (void*)&clMRaster2Camera);
		clStatus |= clSetKernelArg(clK_render, 5, sizeof(cl_mem),  (void*)&clMCamera2World);
		clStatus |= clSetKernelArg(clK_render, 6, sizeof(cl_mem),  (void*)&clMImage);
		CHECK_CL(clStatus);

		clStatus = clEnqueueNDRangeKernel(clQueue, clK_render, 2, NULL, globalWS, localWS, 0, NULL, NULL);
		CHECK_CL(clStatus);
	}

	if(useInterop)
	{
		clStatus = clEnqueueReleaseGLObjects(clQueue, 1, &clMImage, 0, NULL, NULL );
		CHECK_CL(clStatus);
	}

	clStatus = clFinish(clQueue);
	CHECK_CL(clStatus);
}

void COMPCL_PBRTVOLUME::download()
{
	cl_int clStatus = CL_SUCCESS;
	
	const size_t origin[3] = { 0, 0, 0 };
	const size_t region[3] = { launchW, launchH, 1 };
	
	clStatus = clEnqueueReadImage(clQueue, clMImage, CL_TRUE, origin, region, 0, 0, outputFLT, 0 , nullptr, nullptr);
	CHECK_CL(clStatus);

	//savePPM("comp_pbrtvolume_cl", outputFLT, launchW, launchH, 4);
}