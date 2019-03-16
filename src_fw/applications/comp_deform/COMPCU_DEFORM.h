#ifndef COMPCU_DEFORM_H
#define COMPCU_DEFORM_H

#include "COMP_DEFORM.h"

#include "cuutil/cubase.h"

class COMPCU_DEFORM : public COMP_DEFORM
{
public:
	COMPCU_DEFORM();
	~COMPCU_DEFORM();
	
	virtual void init();
	virtual void terminate();
	
	virtual void compute(int currDeform, float currTime);
	virtual void download();
	
public:
	void* cuMVerts;

	cudaGraphicsResource* cuVBOVerts;
};

#endif
