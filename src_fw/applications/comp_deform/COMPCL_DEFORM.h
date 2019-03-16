#ifndef COMPCL_DEFORM_H
#define COMPCL_DEFORM_H

#include "COMP_DEFORM.h"

#include "clutil/clbase.h"

class COMPCL_DEFORM : public COMP_DEFORM
{
public:
	COMPCL_DEFORM();
	~COMPCL_DEFORM();
	
	virtual void init();
	virtual void terminate();
	
	virtual void compute(int currDeform, float currTime);
	virtual void download();
	
public:
	OpenCLUtil cl;

	cl_uint selectedPlatformIndex;
	cl_uint selectedDeviceIndex;

	cl_platform_id selectedPlatformID;
	cl_device_id selectedDeviceID;

	cl_context clContext;
	cl_command_queue clQueue;
	
	cl_program clProg_deform;
	cl_kernel clK_deform;

	cl_mem clMVerts;
};

#endif
