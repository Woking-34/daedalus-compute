#include "appfw/GLUTApplication.h"

#include "math/core/vec.h"
#include "math/core/mat.h"
#include "math/util/camera.h"

#include "assets/meshfile.h"
#include "assets/imagedata.h"

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
	Camera myCamera;
	Vec4f camEye, camCenter, camUp;
	float camFOV, camNear, camFar;

	GLProgram progAlbedoVertCol;

	GLMesh_SOA myMesh;
	GLMesh myMeshWorldGrid;

	std::string seqNameBase, seqNameExt;
	int currID, startID, finishID, filterID;

	int meshSeqSize;
	std::vector<MeshFile> meshSeqVec;
};
