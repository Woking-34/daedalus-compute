#ifndef COMPCU_RTOW_H
#define COMPCU_RTOW_H

#include "COMP_RTOW.h"

#include "cuutil/cubase.h"

class COMPCU_RTOW : public COMP_RTOW
{
public:
	COMPCU_RTOW();
	~COMPCU_RTOW();
	
	void init();
	void terminate();
	
	void compute();
	void download();
	
public:
	cudaGraphicsResource_t viewCudaResource;
	cudaSurfaceObject_t viewCudaSurfaceObject;
	cudaArray_t viewCudaArray;

	void* cuMCamera;
	void* cuMSeed0;
	void* cuMSeed1;
	void* cuMSpheres;
	void* cuMMaterials;
};

#endif
