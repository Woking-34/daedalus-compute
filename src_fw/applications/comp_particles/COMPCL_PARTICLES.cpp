#include "COMPCL_PARTICLES.h"

#include "system/log.h"
#include "system/filesystem.h"

COMPCL_PARTICLES::COMPCL_PARTICLES()
{
	selectedPlatformIndex = -1;
	selectedDeviceIndex = -1;

	clContext = 0;
	clQueue = 0;
	
	clProg_particles = 0;
	clK_reset = 0;
	clK_create = 0;
	clK_collide = 0;
	clK_integrate = 0;

	clMPos = 0;

	clMVel0 = 0;
	clMVel1 = 0;
	clMPList = 0;
	clMHList = 0;
}

COMPCL_PARTICLES::~COMPCL_PARTICLES()
{

}

void COMPCL_PARTICLES::init()
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

	std::string filename = "particles/particles.cl";

	std::string clFileName = FileSystem::GetKernelsCLFolder() + filename;
	std::string clProgStr = loadStr(clFileName);

	const char* clProgramStr = clProgStr.c_str();
	size_t clProgramSize = clProgStr.size();

	clProg_particles = clCreateProgramWithSource(clContext, 1, &clProgramStr, &clProgramSize, &clStatus);
	CHECK_CL(clStatus);

	//clStatus = clBuildProgram(clProg_root, 0, NULL, NULL, NULL, NULL);
	clStatus = clBuildProgram(clProg_particles, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
	if(clStatus == CL_BUILD_PROGRAM_FAILURE)
	{
		size_t buildLogSize = 0;
		clStatus = clGetProgramBuildInfo(clProg_particles, selectedDeviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
		CHECK_CL(clStatus);

		char* buildLog = new char[buildLogSize];
		clStatus = clGetProgramBuildInfo(clProg_particles, selectedDeviceID, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
		CHECK_CL(clStatus);

		std::cout << buildLog << std::endl;
	}

	clK_reset = clCreateKernel(clProg_particles, "resetHeadList", &clStatus);
	CHECK_CL(clStatus);

	clK_create = clCreateKernel(clProg_particles, "createList", &clStatus);
	CHECK_CL(clStatus);

	clK_collide = clCreateKernel(clProg_particles, "collideList", &clStatus);
	CHECK_CL(clStatus);

	clK_integrate = clCreateKernel(clProg_particles, "integrate", &clStatus);
	CHECK_CL(clStatus);
		
	if(useInterop)
	{
		clMPos = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, vboPos, &clStatus);
		CHECK_CL(clStatus);
	}
	else
	{
		clMPos = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, numParticles*sizeof(cl_float4), bufferPos, &clStatus);
		CHECK_CL(clStatus);
	}

	clMVel0 = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, numParticles*sizeof(cl_float4), bufferVel, &clStatus);
	CHECK_CL(clStatus);

	clMVel1 = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, numParticles*sizeof(cl_float4), bufferVel, &clStatus);
	CHECK_CL(clStatus);
	
	clMHList = clCreateBuffer(clContext, CL_MEM_READ_WRITE, numGridCellsPaddded*sizeof(cl_int), NULL, &clStatus);
	CHECK_CL(clStatus);

	clMPList = clCreateBuffer(clContext, CL_MEM_READ_WRITE, numParticles*sizeof(cl_int), NULL, &clStatus);
	CHECK_CL(clStatus);

	clStatus = clFinish(clQueue);
	CHECK_CL(clStatus);	
}

void COMPCL_PARTICLES::terminate()
{
	if (clMPos)
	{
		CHECK_CL(clReleaseMemObject(clMPos));
		clMPos = 0;
	}

	if (clMVel0)
	{
		CHECK_CL(clReleaseMemObject(clMVel0));
		clMVel0 = 0;
	}

	if (clMVel1)
	{
		CHECK_CL(clReleaseMemObject(clMVel1));
		clMVel1 = 0;
	}

	if (clMHList)
	{
		CHECK_CL(clReleaseMemObject(clMHList));
		clMHList = 0;
	}

	if (clMPList)
	{
		CHECK_CL(clReleaseMemObject(clMPList));
		clMPList = 0;
	}
	
	if(clK_reset)
	{
		CHECK_CL(clReleaseKernel(clK_reset));
		clK_reset = 0;
	}
	if (clK_create)
	{
		CHECK_CL(clReleaseKernel(clK_create));
		clK_create = 0;
	}
	if (clK_collide)
	{
		CHECK_CL(clReleaseKernel(clK_collide));
		clK_collide = 0;
	}
	if (clK_integrate)
	{
		CHECK_CL(clReleaseKernel(clK_integrate));
		clK_integrate = 0;
	}

	if(clProg_particles)
	{
		CHECK_CL(clReleaseProgram(clProg_particles));
		clProg_particles = 0;
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

void COMPCL_PARTICLES::compute()
{
	cl_int clStatus = CL_SUCCESS;

	if(useInterop)
	{
		clStatus = clEnqueueAcquireGLObjects(clQueue, 1, &clMPos, 0, NULL, NULL );
		CHECK_CL(clStatus);
	}

	size_t globalWSParticles[3] = { numParticles, 1, 1 };
	size_t globalWSGridCells[3] = { numGridCellsPaddded, 1, 1 };
	size_t localWS[3] = { wgsX, wgsY, wgsZ };
	//size_t* localWS = NULL;

	for (int i = 0; i < 5; ++i)
	{
		{
			clStatus |= clSetKernelArg(clK_reset, 0, sizeof(cl_mem), (void*)&clMHList);
			CHECK_CL(clStatus);

			clStatus = clEnqueueNDRangeKernel(clQueue, clK_reset, 1, NULL, globalWSGridCells, localWS, 0, NULL, NULL);
			CHECK_CL(clStatus);
		}

		{
			clStatus |= clSetKernelArg(clK_create, 0, sizeof(cl_mem), (void*)&clMPos);
			clStatus |= clSetKernelArg(clK_create, 1, sizeof(cl_mem), (void*)&clMHList);
			clStatus |= clSetKernelArg(clK_create, 2, sizeof(cl_mem), (void*)&clMPList);
			clStatus |= clSetKernelArg(clK_create, 3, sizeof(cl_float), (void*)&particleRadius);
			clStatus |= clSetKernelArg(clK_create, 4, sizeof(cl_int), (void*)&gridCells);
			CHECK_CL(clStatus);

			clStatus = clEnqueueNDRangeKernel(clQueue, clK_create, 1, NULL, globalWSParticles, localWS, 0, NULL, NULL);
			CHECK_CL(clStatus);
		}

		{
			clStatus |= clSetKernelArg(clK_collide, 0, sizeof(cl_mem), (void*)&clMHList);
			clStatus |= clSetKernelArg(clK_collide, 1, sizeof(cl_mem), (void*)&clMPList);
			clStatus |= clSetKernelArg(clK_collide, 2, sizeof(cl_mem), (void*)&clMPos);
			clStatus |= clSetKernelArg(clK_collide, 3, sizeof(cl_mem), (void*)&clMVel0);
			clStatus |= clSetKernelArg(clK_collide, 4, sizeof(cl_mem), (void*)&clMVel1);
			clStatus |= clSetKernelArg(clK_collide, 5, sizeof(cl_float), (void*)&particleRadius);
			clStatus |= clSetKernelArg(clK_collide, 6, sizeof(cl_int), (void*)&gridCells);
			CHECK_CL(clStatus);

			clStatus = clEnqueueNDRangeKernel(clQueue, clK_collide, 1, NULL, globalWSParticles, localWS, 0, NULL, NULL);
			CHECK_CL(clStatus);
		}

		{
			clStatus |= clSetKernelArg(clK_integrate, 0, sizeof(cl_mem), (void*)&clMPos);
			clStatus |= clSetKernelArg(clK_integrate, 1, sizeof(cl_mem), (void*)&clMVel0);
			clStatus |= clSetKernelArg(clK_integrate, 2, sizeof(cl_mem), (void*)&clMVel1);
			clStatus |= clSetKernelArg(clK_integrate, 3, sizeof(cl_float), (void*)&particleRadius);
			CHECK_CL(clStatus);

			clStatus = clEnqueueNDRangeKernel(clQueue, clK_integrate, 1, NULL, globalWSParticles, localWS, 0, NULL, NULL);
			CHECK_CL(clStatus);
		}
	}

	if(useInterop)
	{
		clStatus = clEnqueueReleaseGLObjects(clQueue, 1, &clMPos, 0, NULL, NULL );
		CHECK_CL(clStatus);
	}

	clStatus = clFinish(clQueue);
	CHECK_CL(clStatus);
}

void COMPCL_PARTICLES::download()
{
	cl_int clStatus = CL_SUCCESS;
		
	clStatus = clEnqueueReadBuffer(clQueue, clMPos, CL_TRUE, 0, numParticles*sizeof(cl_float4), bufferPos, 0, nullptr, nullptr);
	CHECK_CL(clStatus);
}