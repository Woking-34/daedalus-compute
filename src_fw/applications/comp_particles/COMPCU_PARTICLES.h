#ifndef COMPCU_PARTICLES_H
#define COMPCU_PARTICLES_H

#include "COMP_PARTICLES.h"

#include "cuutil/cubase.h"

class COMPCU_PARTICLES : public COMP_PARTICLES
{
public:
	COMPCU_PARTICLES();
	~COMPCU_PARTICLES();
	
	virtual void init();
	virtual void terminate();
	
	virtual void compute();
	virtual void download();
	
public:
	void* cuMPos;

	void* cuMVel0;
	void* cuMVel1;
	void* cuMHList;
	void* cuMPList;

	cudaGraphicsResource* cuVBOPos;
};

#endif
