#ifndef COMP_PARTICLES_H
#define COMP_PARTICLES_H

#include "system/platform.h"

class COMP_PARTICLES
{
public:
	COMP_PARTICLES();
	~COMP_PARTICLES();

	virtual void init() = 0;
	virtual void terminate() = 0;

	virtual void compute() = 0;
	virtual void download() = 0;
	
public:
	// input data (owner host - interop buffers)
	float* bufferPos;

	// input data (owner host)
	float* bufferCol;
	float* bufferVel;
	
	bool useInterop;
	unsigned int vboPos;
	unsigned int vboCol;
	
	int wgsX, wgsY, wgsZ;
	
	bool useCLGDevice;
	int useCLPId, useCLDId; 

	int numParticles;
	int numParticlesDimX;
	int numParticlesDimY;
	int numParticlesDimZ;

	float particleRadius;

	int gridCells;
	int numGridCells;
	int numGridCellsPaddded;
};

#endif
