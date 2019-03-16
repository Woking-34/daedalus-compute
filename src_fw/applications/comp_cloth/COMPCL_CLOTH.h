#ifndef COMPCL_CLOTH_H
#define COMPCL_CLOTH_H

#include "COMP_CLOTH.h"

#include "clutil/clbase.h"

class COMPCL_CLOTH : public COMP_CLOTH
{
public:
	COMPCL_CLOTH();
	~COMPCL_CLOTH();
	
	virtual void init();
	virtual void terminate();
	
	virtual void compute();
	virtual void download();
	
public:
	OpenCLUtil cl;

	cl_uint selectedPlatformIndex;
	cl_uint selectedDeviceIndex;

	cl_platform_id selectedPlatformID;
	cl_device_id selectedDeviceID;

	cl_context clContext;
	cl_command_queue clQueue;
	
	cl_program clProg_cloth;
	cl_kernel clK_cloth;

	cl_mem clMMass;

	cl_mem clMIn;
	cl_mem clMInOld;
	cl_mem clMOut;
	cl_mem clMOutOld;
	cl_mem clMNormals;
};

#endif
