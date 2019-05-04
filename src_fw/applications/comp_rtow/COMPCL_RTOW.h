#ifndef COMPCL_RTOW_H
#define COMPCL_RTOW_H

#include "COMP_RTOW.h"

#include "clutil/clbase.h"

class COMPCL_RTOW : public COMP_RTOW
{
public:
	COMPCL_RTOW();
	~COMPCL_RTOW();
	
	void init();
	void terminate();
	
	void compute();
	void download();
	
public:
	OpenCLUtil cl;

	cl_uint selectedPlatformIndex;
	cl_uint selectedDeviceIndex;

	cl_platform_id selectedPlatformID;
	cl_device_id selectedDeviceID;

	cl_context clContext;
	cl_command_queue clQueue;
	
	cl_program clProg_rtow;
	cl_kernel clK_render;

	cl_mem clMImage;
	cl_mem clMCamera;
	cl_mem clMSeed0;
	cl_mem clMSeed1;
	cl_mem clMSpheres;
	cl_mem clMMaterials;
};

#endif
