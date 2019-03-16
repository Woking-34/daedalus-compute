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

class COMP_PARTICLES;

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

	int wgsX, wgsY, wgsZ;

	int numParticles;

	int numParticlesDimX;
	int numParticlesDimY;
	int numParticlesDimZ;

	float particleRadius;

	int gridCells;
	int numGridCells;
	int numGridCellsPaddded;

	COMP_PARTICLES* calculator;

	GLuint vboPos;
	GLuint vboCol;

	GLuint vboBBox;

	Vec4f* initPos;
	Vec4f* initCol;
	Vec4f* initVel;
	
	GLProgram renderProgram_BBox;
	GLProgram renderProgram_Particles;
};
