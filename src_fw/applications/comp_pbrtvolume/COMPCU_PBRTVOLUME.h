#ifndef COMPCU_RTOW_H
#define COMPCU_RTOW_H

#include "COMP_PBRTVOLUME.h"

#include "cuutil/cubase.h"

class COMPCU_PBRTVOLUME : public COMP_PBRTVOLUME
{
public:
	COMPCU_PBRTVOLUME();
	~COMPCU_PBRTVOLUME();
	
	void init();
	void terminate();
	
	void compute();
	void download();
	
public:
	cudaGraphicsResource_t viewCudaResource;
	cudaSurfaceObject_t viewCudaSurfaceObject;
	cudaArray_t viewCudaArray;

	void* cuMCamera;
	void* cuMVolumeData;
};

#endif
