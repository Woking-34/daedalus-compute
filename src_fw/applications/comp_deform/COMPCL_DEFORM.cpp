#include "COMPCL_DEFORM.h"

#include "system/log.h"
#include "system/filesystem.h"

COMPCL_DEFORM::COMPCL_DEFORM()
{
	selectedPlatformIndex = -1;
	selectedDeviceIndex = -1;

	clContext = 0;
	clQueue = 0;
	
	clProg_deform = 0;
	clK_deform = 0;

	clMVerts = 0;
}

COMPCL_DEFORM::~COMPCL_DEFORM()
{

}

void COMPCL_DEFORM::init()
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

	std::string filename = "deform/deform.cl";

	std::string clFileName = FileSystem::GetKernelsCLFolder() + filename;
	std::string clProgStr = loadStr(clFileName);

	const char* clProgramStr = clProgStr.c_str();
	size_t clProgramSize = clProgStr.size();

	clProg_deform = clCreateProgramWithSource(clContext, 1, &clProgramStr, &clProgramSize, &clStatus);
	CHECK_CL(clStatus);

	clStatus = clBuildProgram(clProg_deform, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
	if(clStatus == CL_BUILD_PROGRAM_FAILURE)
	{
		size_t buildLogSize = 0;
		clStatus = clGetProgramBuildInfo(clProg_deform, selectedDeviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
		CHECK_CL(clStatus);

		char* buildLog = new char[buildLogSize];
		clStatus = clGetProgramBuildInfo(clProg_deform, selectedDeviceID, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
		CHECK_CL(clStatus);

		std::cout << buildLog << std::endl;
	}

	clK_deform = clCreateKernel(clProg_deform, "deform", &clStatus);
	CHECK_CL(clStatus);
		
	if(useInterop)
	{
		clMVerts = clCreateFromGLBuffer(clContext, CL_MEM_WRITE_ONLY, vbo, &clStatus);
		CHECK_CL(clStatus);
	}
	else
	{
		clMVerts = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, 2*launchW*launchH*sizeof(cl_float4), bufferVertices, &clStatus);
		CHECK_CL(clStatus);
	}

	clStatus = clFinish(clQueue);
	CHECK_CL(clStatus);	
}

void COMPCL_DEFORM::terminate()
{
	if (clMVerts)
	{
		CHECK_CL(clReleaseMemObject(clMVerts));
		clMVerts = 0;
	}
	
	if(clK_deform)
	{
		CHECK_CL(clReleaseKernel(clK_deform));
		clK_deform = 0;
	}

	if(clProg_deform)
	{
		CHECK_CL(clReleaseProgram(clProg_deform));
		clProg_deform = 0;
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

void COMPCL_DEFORM::compute(int currDeform, float currTime)
{
	cl_int clStatus = CL_SUCCESS;

	if(useInterop)
	{
		clStatus = clEnqueueAcquireGLObjects(clQueue, 1, &clMVerts, 0, NULL, NULL );
		CHECK_CL(clStatus);
	}

	size_t globalWS[3] = { launchW, launchH, 1 };
	size_t localWS[3] = { wgsX, wgsY, 1 };
	//size_t* localWS = NULL;

	{		
		clStatus |= clSetKernelArg(clK_deform,  0, sizeof(cl_mem), (void*)&clMVerts);
		clStatus |= clSetKernelArg(clK_deform,  1, sizeof(cl_int), &currDeform);
		clStatus |= clSetKernelArg(clK_deform,  2, sizeof(cl_float), &currTime);
		clStatus |= clSetKernelArg(clK_deform,  3, sizeof(cl_float), &sizeX);
		clStatus |= clSetKernelArg(clK_deform,  4, sizeof(cl_float), &stepX);
		clStatus |= clSetKernelArg(clK_deform,  5, sizeof(cl_float), &sizeY);
		clStatus |= clSetKernelArg(clK_deform,  6, sizeof(cl_float), &stepY);
		CHECK_CL(clStatus);
		
		clStatus = clEnqueueNDRangeKernel(clQueue, clK_deform, 2, NULL, globalWS, localWS, 0, NULL, NULL);
		CHECK_CL(clStatus);
	}

	if(useInterop)
	{
		clStatus = clEnqueueReleaseGLObjects(clQueue, 1, &clMVerts, 0, NULL, NULL );
		CHECK_CL(clStatus);
	}

	clStatus = clFinish(clQueue);
	CHECK_CL(clStatus);
}

void COMPCL_DEFORM::download()
{
	cl_int clStatus = CL_SUCCESS;
		
	clStatus = clEnqueueReadBuffer(clQueue, clMVerts, CL_TRUE, 0, 2*launchW*launchH*sizeof(cl_float4), bufferVertices, 0, nullptr, nullptr);
	CHECK_CL(clStatus);
}