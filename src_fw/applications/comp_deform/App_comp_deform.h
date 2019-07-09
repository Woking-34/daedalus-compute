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

class COMP_DEFORM;

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
	int deformMode;

	daedalus::Camera mainCamera;

	// run params
	bool useInterop;

	// codepath select str
	std::string api;

	// cl platform/device select
	bool useCLGDevice;
	int useCLPId, useCLDId;

	int launchW, launchH;
	int wgsX, wgsY;

	float sizeX, sizeY;
	float stepX, stepY;

	COMP_DEFORM* calculator;

	GLuint vboID;
	GLuint iboID;

	daedalus::Vec4f* initVBO;
	unsigned int* initIBO;
	
	GLProgram renderProgram;
};
