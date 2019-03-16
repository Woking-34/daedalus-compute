#ifndef COMPGL_CLOTH_H
#define COMPGL_CLOTH_H

#include "COMP_CLOTH.h"

#include "glutil/glbase.h"
#include "glutil/glprogram.h"

class COMPGL_CLOTH : public COMP_CLOTH
{
public:
	COMPGL_CLOTH();
	~COMPGL_CLOTH();

	virtual void init();
	virtual void terminate();

	virtual void compute();
	virtual void download();
	
public:	
	GLProgram clothProgram;

	GLuint massSSBO;
};

#endif
