#include "COMPCL_CLOTH.h"

#include "system/log.h"
#include "system/filesystem.h"

COMPCL_CLOTH::COMPCL_CLOTH()
{
	selectedPlatformIndex = -1;
	selectedDeviceIndex = -1;

	clContext = 0;
	clQueue = 0;
	
	clProg_cloth = 0;
	clK_cloth = 0;

	clMMass = 0;

	clMIn = 0;
	clMInOld = 0;
	clMOut = 0;
	clMOutOld = 0;
	clMNormals = 0;
}

COMPCL_CLOTH::~COMPCL_CLOTH()
{

}

void COMPCL_CLOTH::init()
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

	if(useCLPId != -1 && selectedDeviceIndex != -1)
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

	clQueue = clCreateCommandQueue( clContext, selectedDeviceID, 0, &clStatus );
	CHECK_CL(clStatus);

	std::string filename = "cloth/cloth.cl";

	std::string clFileName = FileSystem::GetKernelsCLFolder() + filename;
	std::string clProgStr = loadStr(clFileName);

	const char* clProgramStr = clProgStr.c_str();
	size_t clProgramSize = clProgStr.size();

	clProg_cloth = clCreateProgramWithSource(clContext, 1, &clProgramStr, &clProgramSize, &clStatus);
	CHECK_CL(clStatus);

	//clStatus = clBuildProgram(clProg_root, 0, NULL, NULL, NULL, NULL);
	clStatus = clBuildProgram(clProg_cloth, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
	if(clStatus == CL_BUILD_PROGRAM_FAILURE)
	{
		size_t buildLogSize = 0;
		clStatus = clGetProgramBuildInfo(clProg_cloth, selectedDeviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
		CHECK_CL(clStatus);

		char* buildLog = new char[buildLogSize];
		clStatus = clGetProgramBuildInfo(clProg_cloth, selectedDeviceID, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
		CHECK_CL(clStatus);

		std::cout << buildLog << std::endl;
	}

	clK_cloth = clCreateKernel(clProg_cloth, "verlet", &clStatus);
	CHECK_CL(clStatus);
		
	if(useInterop)
	{
		clMIn = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, vboInCurr, &clStatus);
		CHECK_CL(clStatus);

		clMInOld = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, vboInPrev, &clStatus);
		CHECK_CL(clStatus);

		clMOut = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, vboOutCurr, &clStatus);
		CHECK_CL(clStatus);

		clMOutOld = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, vboOutPrev, &clStatus);
		CHECK_CL(clStatus);

		clMNormals = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, vboNormals, &clStatus);
		CHECK_CL(clStatus);
	}
	else
	{
		clMIn = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, launchW*launchH*sizeof(cl_float4), bufferPositions, &clStatus);
		CHECK_CL(clStatus);

		clMInOld = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, launchW*launchH*sizeof(cl_float4), bufferPositions, &clStatus);
		CHECK_CL(clStatus);

		clMOut = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, launchW*launchH*sizeof(cl_float4), bufferPositions, &clStatus);
		CHECK_CL(clStatus);

		clMOutOld = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, launchW*launchH*sizeof(cl_float4), bufferPositions, &clStatus);
		CHECK_CL(clStatus);

		clMNormals = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, launchW*launchH*sizeof(cl_float4), bufferNormals, &clStatus);
		CHECK_CL(clStatus);
	}

	clMMass = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, launchW*launchH*sizeof(float), bufferWeights, &clStatus);
	CHECK_CL(clStatus);

	clStatus = clFinish(clQueue);
	CHECK_CL(clStatus);	
}

void COMPCL_CLOTH::terminate()
{
	if (clMMass)
	{
		CHECK_CL(clReleaseMemObject(clMMass));
		clMMass = 0;
	}

	if (clMIn)
	{
		CHECK_CL(clReleaseMemObject(clMIn));
		clMIn = 0;
	}

	if (clMInOld)
	{
		CHECK_CL(clReleaseMemObject(clMInOld));
		clMInOld = 0;
	}

	if (clMOut)
	{
		CHECK_CL(clReleaseMemObject(clMOut));
		clMOut = 0;
	}

	if (clMOutOld)
	{
		CHECK_CL(clReleaseMemObject(clMOutOld));
		clMOutOld = 0;
	}

	if (clMNormals)
	{
		CHECK_CL(clReleaseMemObject(clMNormals));
		clMNormals = 0;
	}
	
	if(clK_cloth)
	{
		CHECK_CL(clReleaseKernel(clK_cloth));
		clK_cloth = 0;
	}

	if(clProg_cloth)
	{
		CHECK_CL(clReleaseProgram(clProg_cloth));
		clProg_cloth = 0;
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

void COMPCL_CLOTH::compute()
{
	cl_int clStatus = CL_SUCCESS;

	if(useInterop)
	{
		clStatus = clEnqueueAcquireGLObjects(clQueue, 1, &clMIn, 0, NULL, NULL );
		CHECK_CL(clStatus);

		clStatus = clEnqueueAcquireGLObjects(clQueue, 1, &clMInOld, 0, NULL, NULL);
		CHECK_CL(clStatus);

		clStatus = clEnqueueAcquireGLObjects(clQueue, 1, &clMOut, 0, NULL, NULL);
		CHECK_CL(clStatus);

		clStatus = clEnqueueAcquireGLObjects(clQueue, 1, &clMOutOld, 0, NULL, NULL);
		CHECK_CL(clStatus);

		clStatus = clEnqueueAcquireGLObjects(clQueue, 1, &clMNormals, 0, NULL, NULL);
		CHECK_CL(clStatus);
	}

	size_t globalWS[3] = { launchW, launchH, 1 };
	size_t localWS[3] = { wgsX, wgsY, 1 };
	//size_t* localWS = NULL;

	{		
		clStatus |= clSetKernelArg(clK_cloth,  0, sizeof(cl_mem), (void *)&clMIn);
		clStatus |= clSetKernelArg(clK_cloth,  1, sizeof(cl_mem), (void *)&clMInOld);
		clStatus |= clSetKernelArg(clK_cloth,  2, sizeof(cl_mem), (void *)&clMOut);
		clStatus |= clSetKernelArg(clK_cloth,  3, sizeof(cl_mem), (void *)&clMOutOld);
		clStatus |= clSetKernelArg(clK_cloth,  4, sizeof(cl_mem), (void *)&clMNormals);
		clStatus |= clSetKernelArg(clK_cloth,  5, sizeof(cl_mem), (void *)&clMMass);

		clStatus |= clSetKernelArg(clK_cloth,  6, sizeof(cl_float), &damp);
		clStatus |= clSetKernelArg(clK_cloth,  7, sizeof(cl_float), &dt);
		clStatus |= clSetKernelArg(clK_cloth,  8, sizeof(cl_float), &stepX);
		clStatus |= clSetKernelArg(clK_cloth,  9, sizeof(cl_float), &stepY);
		CHECK_CL(clStatus);

		clStatus = clEnqueueNDRangeKernel(clQueue, clK_cloth, 2, NULL, globalWS, localWS, 0, NULL, NULL);
		CHECK_CL(clStatus);
	}

	if(useInterop)
	{
		clStatus = clEnqueueReleaseGLObjects(clQueue, 1, &clMIn, 0, NULL, NULL );
		CHECK_CL(clStatus);

		clStatus = clEnqueueReleaseGLObjects(clQueue, 1, &clMInOld, 0, NULL, NULL);
		CHECK_CL(clStatus);

		clStatus = clEnqueueReleaseGLObjects(clQueue, 1, &clMOut, 0, NULL, NULL);
		CHECK_CL(clStatus);

		clStatus = clEnqueueReleaseGLObjects(clQueue, 1, &clMOutOld, 0, NULL, NULL);
		CHECK_CL(clStatus);

		clStatus = clEnqueueReleaseGLObjects(clQueue, 1, &clMNormals, 0, NULL, NULL);
		CHECK_CL(clStatus);
	}

	clStatus = clFinish(clQueue);
	CHECK_CL(clStatus);

	std::swap(clMIn, clMOut);
	std::swap(clMInOld, clMOutOld);

	std::swap(vboInCurr, vboOutCurr);
	std::swap(vboInPrev, vboOutPrev);
}

void COMPCL_CLOTH::download()
{
	cl_int clStatus = CL_SUCCESS;
		
	clStatus = clEnqueueReadBuffer(clQueue, clMIn, CL_TRUE, 0, launchW*launchH*sizeof(cl_float4), bufferPositions, 0, nullptr, nullptr);
	CHECK_CL(clStatus);

	clStatus = clEnqueueReadBuffer(clQueue, clMNormals, CL_TRUE, 0, launchW*launchH*sizeof(cl_float4), bufferNormals, 0, nullptr, nullptr);
	CHECK_CL(clStatus);
}