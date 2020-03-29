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
#include "glutil/glframebuffer.h"

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

	bool isSpotInput;
	daedalus::Camera mainCamera;

	GLProgram progAlbedo, progAlbedoTex;
	GLProgram progLightSpot, progLightSpotShadow;
	
	GLMesh meshGrid;
	GLMesh meshBox;
	GLMesh meshSphere;

	float range1, range2, angle1, angle2;
	daedalus::Vec4f posSource_SpotLight, posTarget_SpotLight;
	GLMesh meshConeWire;

	GLuint texIdCrate;
	GLuint texIdChecker;

	GLFrameBuffer fboShadow;
	GLsizei shadowSizeX, shadowSizeY;

	void RenderDebug();

	void RenderAlbedo(const daedalus::Mat44f& viewMat, const daedalus::Mat44f& projMat);

	void RenderShaded();
	void RenderShadedShadow();
};
