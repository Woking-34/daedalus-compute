#include "App_gl_mesh_normal.h"

#include "system/log.h"
#include "system/filesystem.h"

using namespace daedalus;

App AppInstance;

App::App()
{
	currWidth = 800;
	currHeight = 800;

	float scaleCamDist = 3.75f;

	camEye = Vec4f(scaleCamDist, scaleCamDist, scaleCamDist, 1.0f);
	camCenter =  Vec4f(0.0f, 0.0f, 0.0f, 1.0f);
	camUp = Vec4f(0.0f, 1.0f, 0.0f, 0.0f);

    //camEye = Vec4f(0.0f, 8.0f, -2.0f, 1.0f);
    //camCenter = Vec4f(0.0f, 0.0f, 0.0f, 1.0f);
    //camUp = Vec4f(0.0f, 1.0f, 0.0f, 0.0f);

	camFOV = 60.0f;
	camNear = 0.25f;
	camFar = 100.0f;
}

App::~App()
{

}

void App::Initialize()
{
	mainCamera.setViewParams( camEye, camCenter, camUp );
	mainCamera.setProjParams( camFOV, (float)(currWidth)/(float)(currHeight), camNear, camFar );
	mainCamera.updateRays();

	appCamera = &mainCamera;

	{
		MeshFile myMeshFileGrid;
		myMeshFileGrid.createGridTris(4.0f, 4.0f, 8, 8);
		myMeshFileGrid.setColorVec(1.0f, 1.0f, 1.0f);
		myMeshFileGrid.flattenIndices();

		MeshFile myMeshFileSphere;
		myMeshFileSphere.createSphereTris(0.75f, 16, 16);
		myMeshFileSphere.setColorVec(1.0f, 1.0f, 1.0f);
		myMeshFileSphere.flattenIndices();

		myMeshGrid.create( myMeshFileGrid );
		myMeshSphere.create( myMeshFileSphere );
	}

	progAlbedo.addFile("albedo_vertcolor.vert", GL_VERTEX_SHADER);
	progAlbedo.addFile("albedo_vertcolor.frag", GL_FRAGMENT_SHADER);
	progAlbedo.buildProgram();

	progNormViz.addFile("normviz.vert", GL_VERTEX_SHADER);
	progNormViz.addFile("normviz.geom", GL_GEOMETRY_SHADER);
	progNormViz.addFile("normviz.frag", GL_FRAGMENT_SHADER);
	progNormViz.buildProgram();

	// init gl state
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);	
}

void App::Update()
{
}

void App::Render()
{
	mainCamera.setProjParams( 60.0f, (float)(currWidth)/(float)(currHeight), 0.25f, 100.0f );
	mainCamera.updateRays();

	glViewport(0, 0, currWidth, currHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (1)
	{
		progAlbedo.useProgram();

        Mat44f mat = createTranslationMatrix(0.0f, 0.0f, 0.0f);

        progAlbedo.setFloatMatrix44(mat * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat");
		myMeshGrid.render( progAlbedo.getAttribLocations() );

		progAlbedo.setFloatMatrix44( createTranslationMatrix(0.0f, 2.0f, 0.0f) * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );
		myMeshSphere.render( progAlbedo.getAttribLocations() );

		glUseProgram(0);
	}

    if (1)
	{
		Mat44f modelMat = Mat44f(one);

		progNormViz.useProgram();
		
		progNormViz.setFloatValue(0.25f, "length");

		progNormViz.setFloatMatrix44(mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );
		myMeshGrid.render( progNormViz.getAttribLocations() );

		progNormViz.setFloatMatrix44( createTranslationMatrix(0.0f, 2.0f, 0.0f) * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );
		myMeshSphere.render( progNormViz.getAttribLocations() );

		glUseProgram(0);
	}
}

void App::Terminate()
{
	progAlbedo.clear();
	progNormViz.clear();

	myMeshGrid.clear();
	myMeshSphere.clear();

	CHECK_GL;
}

std::string App::GetName()
{
	return std::string("gl_mesh_normal");
}