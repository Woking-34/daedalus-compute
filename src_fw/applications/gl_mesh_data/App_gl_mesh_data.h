#include "appfw/GLUTApplication.h"

#include "math/core/vec.h"
#include "math/core/mat.h"
#include "math/util/camera.h"

#include "assets/meshfile.h"
#include "assets/imagedata.h"

#include "glutil/glprogram.h"
#include "glutil/glmesh.h"

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

	const int numScenes;
	int sceneSelcet[3];
	Vec4f camInitPos[3];

	GLProgram progAlbedoVertCol;
	GLProgram progAlbedoTexCol;

	// scene with glasses
	GLMesh myMesh;
	GLMesh myMeshWorldAABB;
	GLMesh myMeshWorldGrid;
	
	// scene with poolballs
	GLMesh meshSpherePool;
	GLMesh meshGridPool;

	// cornell box from quads
	GLMesh meshCornell; // tesselated quads
	void createCornellBox(MeshFile& mfCornell);

	GLuint texIds[16]; // 15 ball + cloth
	Mat44f ballMatrices[15];

	float ballRad;
	float clothSize;
};
