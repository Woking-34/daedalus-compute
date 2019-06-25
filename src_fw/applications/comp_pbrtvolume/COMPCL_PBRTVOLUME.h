#ifndef COMPCL_PBRTVOLUME_H
#define COMPCL_PBRTVOLUME_H

#include "COMP_PBRTVOLUME.h"

#include "clutil/clbase.h"

class COMPCL_PBRTVOLUME : public COMP_PBRTVOLUME
{
public:
	COMPCL_PBRTVOLUME();
	~COMPCL_PBRTVOLUME();
	
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
	
	cl_program clProg_pbrtvolume;
	cl_kernel clK_render;

	cl_mem clMImage;
	cl_mem clMCamera;
	cl_mem clMVolumeData;
};

#endif
