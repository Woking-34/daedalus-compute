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

class App : public GLUTApplication
{

public:
	App();
	~App();
	
public:
	virtual void Initialize();
	virtual void Update();
	virtual void Render();
	virtual void Terminate();

	virtual std::string GetName();

protected:
	int renderMode;

	Camera mainCamera;

	VolumeData< uchar > volumeData;
	//VolumeData< float > volumeData;
	Vec4uc* volumeDataTFF;

	std::string volumeName;
	GLsizei volumeW, volumeH, volumeD;
	GLsizei volumeWPadded, volumeHPadded, volumeDPadded;
	float isoLevel, isoCurr;

	GLsizei actualW, actualH, actualD; 

	// gl texture 3d for volume data
	GLuint volumeTexId;

	// gl texture 1d for tff data
	GLuint tffTexId;

	// gl programs
	GLProgram albedoProgram;
	GLProgram volumeProgramMC;
	GLProgram volumeProgramRC;
	GLProgram volumeProgramRCTFF;
	
	// gl textures for marching cubes tri indexing data
	GLuint cubeFlagsTexId;
	GLuint triTableTexId;

	// gl point grid mesh for marching cubes gs
	GLMesh volumeMeshMCGS;

	// gl mesh for raycast FSQ
	GLMesh volumeMeshFSQRC;

	// gl mesg for debug helper geom
	GLMesh debugBox;
	GLMesh debugAxis;
};