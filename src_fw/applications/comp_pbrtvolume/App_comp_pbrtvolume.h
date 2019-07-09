#include "appfw/GLUTApplication.h"

#include "math/core/vec.h"
#include "math/core/mat.h"
#include "math/util/camera.h"

#include "assets/meshfile.h"
#include "assets/imagedata.h"
#include "assets/volumedata.h"

#include "glutil/glprogram.h"
#include "glutil/glmesh.h"
#include "glutil/glmesh_soa.h"

class COMP_PBRTVOLUME;

class App : public GLUTApplication
{
public:
	App();
	~App();

public:
	virtual void PrintCommandLineHelp();

	virtual void Initialize();
	virtual void Update();
	virtual void Render();
	virtual void Terminate();

	virtual std::string GetName();

protected:
	bool isDynamicCamera;
	daedalus::Camera mainCamera;

	// run params
	bool useInterop;

	// codepath select str
	std::string api;

	// cl platform/device select
	bool useCLGDevice;
	int useCLPId, useCLDId;

	int launchW, launchH;
	int wgsX, wgsY, wgsZ;

	float hostCameraArray[8*4];
	float* volumeData;
	std::vector<float> raster2camera;
	std::vector<float> camera2world;

	int vX, vY, vZ;

	float* outputFLT;

	COMP_PBRTVOLUME* calculator;

	GLuint texId;
	
	GLMesh fsqMesh;
	GLProgram fsqProgram;

	/// camera params
	daedalus::Vec4f origin, u, v, w;
	float vfov = 20.0f;
	float lens_radius;
	
	void initScene();
	void initCamera();
	void updateCameraArray();
};
