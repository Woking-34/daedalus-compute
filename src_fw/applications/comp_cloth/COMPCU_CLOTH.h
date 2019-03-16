#ifndef COMPCU_CLOTH_H
#define COMPCU_CLOTH_H

#include "COMP_CLOTH.h"

#include "cuutil/cubase.h"

class COMPCU_CLOTH : public COMP_CLOTH
{
public:
	COMPCU_CLOTH();
	~COMPCU_CLOTH();
	
	virtual void init();
	virtual void terminate();
	
	virtual void compute();
	virtual void download();
	
public:
	void* cuMMass;

	void* cuMIn;
	void* cuMInOld;
	void* cuMOut;
	void* cuMOutOld;
	void* cuMNormals;

	cudaGraphicsResource* cuVBOIn;
	cudaGraphicsResource* cuVBOInOld;
	cudaGraphicsResource* cuVBOOut;
	cudaGraphicsResource* cuVBOOutOld;
	cudaGraphicsResource* cuVBONormals;
};

#endif
