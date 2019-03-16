#ifndef COMPGL_PARTICLES_H
#define COMPGL_PARTICLES_H

#include "COMP_PARTICLES.h"

#include "glutil/glbase.h"
#include "glutil/glprogram.h"

class COMPGL_PARTICLES : public COMP_PARTICLES
{
public:
	COMPGL_PARTICLES();
	~COMPGL_PARTICLES();

	virtual void init();
	virtual void terminate();

	virtual void compute();
	virtual void download();
	
public:	
	GLProgram computeProgram_reset;
	GLProgram computeProgram_create;
	GLProgram computeProgram_collide;
	GLProgram computeProgram_integrate;

	GLuint vel0SSBO, vel1SSBO;
	GLuint hListSSBO, pListSSBO;
};

#endif
