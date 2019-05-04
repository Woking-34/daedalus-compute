#ifndef COMP_RTOW_H
#define COMP_RTOW_H

#include "system/platform.h"

struct rtow_sphere;
struct rtow_material;

class COMP_RTOW
{
public:
	COMP_RTOW();
	~COMP_RTOW();

	virtual void init() = 0;
	virtual void terminate() = 0;

	virtual void compute() = 0;
	virtual void download() = 0;
	
public:
	// input data (owner host)
	float* cameraArray;
	unsigned int* seed0;
	unsigned int* seed1;

	bool isDynamicCamera;

	int sphereNum;
	rtow_sphere* sphereArrayHost;
	int materialNum;
	rtow_material* materialArrayHost;

	// output data (owner host - interop tex 2d ptr)
	float* outputFLT;
	
	bool useInterop;
	unsigned int interopId;

	unsigned int sampleNum;
	unsigned int launchW, launchH;
	unsigned int wgsX, wgsY, wgsZ;
	
	bool useCLGDevice;
	int useCLPId, useCLDId; 
};

#endif
