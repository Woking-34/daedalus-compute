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

class COMP_CLOTH;

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
	// run params
	bool useInterop;

	// codepath select str
	std::string api;

	// cl platform/device select
	bool useCLGDevice;
	int useCLPId, useCLDId;

	Camera mainCamera;

	int launchW, launchH;
	int wgsX, wgsY;

	float height;
	float sizeX, sizeY;
	float stepX, stepY;
	float mass, damp, dt;

	COMP_CLOTH* calculator;

	GLuint vboID_InCurr;
	GLuint vboID_InPrev;
	GLuint vboID_OutCurr;
	GLuint vboID_OutPrev;
	GLuint vboID_Normals;
	GLuint iboID;

	Vec4f* initPositions;
	Vec4f* initNormals;
	unsigned int* initIBO;
	
	float* initWeights;

	GLProgram renderProgram_Velvet;
};
