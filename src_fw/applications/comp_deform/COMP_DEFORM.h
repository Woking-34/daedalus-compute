#ifndef COMP_DEFORM_H
#define COMP_DEFORM_H

#include "system/platform.h"

class COMP_DEFORM
{
public:
	COMP_DEFORM();
	~COMP_DEFORM();

	virtual void init() = 0;
	virtual void terminate() = 0;

	virtual void compute(int currDeform, float currTime) = 0;
	virtual void download() = 0;
	
public:
	// input data (owner host - interop buffers host)
	float* bufferVertices;

	bool useInterop;
	unsigned int vbo;

	size_t launchW, launchH;
	size_t wgsX, wgsY;
	
	bool useCLGDevice;
	int useCLPId, useCLDId; 

	float sizeX, sizeY;
	float stepX, stepY;
};

#endif
