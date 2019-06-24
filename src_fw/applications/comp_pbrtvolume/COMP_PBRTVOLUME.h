#ifndef COMP_PBRTVOLUME_H
#define COMP_PBRTVOLUME_H

#include "system/platform.h"

struct rtow_sphere;
struct rtow_material;

class COMP_PBRTVOLUME
{
public:
	COMP_PBRTVOLUME();
	~COMP_PBRTVOLUME();

	virtual void init() = 0;
	virtual void terminate() = 0;

	virtual void compute() = 0;
	virtual void download() = 0;
	
public:
	// input data (owner host)
	float* cameraArray;
	float* volumeData;
	float* raster2camera;
	float* camera2world;

	int vX, vY, vZ;

	bool isDynamicCamera;

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
