#include "appfw/GLUTApplication.h"

#include "math/core/vec.h"
#include "math/core/mat.h"
#include "math/util/camera.h"

#include "assets/meshfile.h"

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
	daedalus::Camera mainCamera;
	daedalus::Vec4f camEye, camCenter, camUp;
	float camFOV, camNear, camFar;

	GLProgram progAlbedo;
	GLProgram progNormViz;

	GLMesh myMeshGrid;
	GLMesh myMeshSphere;
};
