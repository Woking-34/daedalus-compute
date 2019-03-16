#ifndef COMP_CLOTH_H
#define COMP_CLOTH_H

#include "system/platform.h"

class COMP_CLOTH
{
public:
	COMP_CLOTH();
	~COMP_CLOTH();

	virtual void init() = 0;
	virtual void terminate() = 0;

	virtual void compute() = 0;
	virtual void download() = 0;
	
public:
	// input data (owner host - interop buffers host)
	float* bufferPositions;
	float* bufferNormals;

	// input data (owner host - ball weights)
	float* bufferWeights;
	
	bool useInterop;
	unsigned int vboInCurr;
	unsigned int vboInPrev;
	unsigned int vboOutCurr;
	unsigned int vboOutPrev;
	unsigned int vboNormals;

	int launchW, launchH;
	int wgsX, wgsY;
	
	bool useCLGDevice;
	int useCLPId, useCLDId; 

	float sizeX, sizeY;
	float stepX, stepY;
	float mass, damp, dt;
};

#endif
