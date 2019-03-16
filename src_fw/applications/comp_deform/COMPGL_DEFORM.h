#ifndef COMPGL_DEFORM_H
#define COMPGL_DEFORM_H

#include "COMP_DEFORM.h"

#include "glutil/glbase.h"
#include "glutil/glprogram.h"

class COMPGL_DEFORM : public COMP_DEFORM
{
public:
	COMPGL_DEFORM();
	~COMPGL_DEFORM();

	virtual void init();
	virtual void terminate();

	virtual void compute(int currDeform, float currTime);
	virtual void download();
	
public:	
	GLProgram deformProgram;
};

#endif
