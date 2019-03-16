#ifndef COMPCL_PARTICLES_H
#define COMPCL_PARTICLES_H

#include "COMP_PARTICLES.h"

#include "clutil/clbase.h"

class COMPCL_PARTICLES : public COMP_PARTICLES
{
public:
	COMPCL_PARTICLES();
	~COMPCL_PARTICLES();
	
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
	
	cl_program clProg_particles;
	cl_kernel clK_reset;
	cl_kernel clK_create;
	cl_kernel clK_collide;
	cl_kernel clK_integrate;

	cl_mem clMPos;

	cl_mem clMVel0, clMVel1;
	cl_mem clMHList, clMPList;
};

#endif
