#include "App_gl_mesh_seq.h"

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
	camCenter = Vec4f(0.0f, 0.0f, 0.0f, 1.0f);
	camUp = Vec4f(0.0f, 1.0f, 0.0f, 0.0f);

	camFOV = 60.0f;
	camNear = 0.25f;
	camFar = 100.0f;

	seqNameBase = "cloth_ball";
	seqNameExt = ".ply";

	startID = 0;
	finishID = 93;
	filterID = 5;

	currID = 0;
}

App::~App()
{

}

void App::Initialize()
{
	myCamera.setViewParams( camEye, camCenter, camUp );
	myCamera.setProjParams( camFOV, (float)(currWidth)/(float)(currHeight), camNear, camFar );
	myCamera.updateRays();
	
	for(int i = startID; i <= finishID; i+=filterID)
	{
		std::stringstream fileNameSS;
		fileNameSS << seqNameBase << i << seqNameExt;

		MeshFile myMeshFile;
		myMeshFile.setName(fileNameSS.str(), "assets/cloth_ball_plys/");
		myMeshFile.loadFromFile();
		
		myMeshFile.generateNormals();
		myMeshFile.copyNormalsToColors();

		myMeshFile.scale(0.1f);

		myMeshFile.flattenIndices();

		meshSeqVec.push_back(myMeshFile);
	}

	meshSeqSize = (int)meshSeqVec.size();

	{
		myMesh.createIBO(meshSeqVec[0].numPrimitives, &meshSeqVec[0].indicesVecFlat[0]);
		myMesh.createVBO(0, meshSeqVec[0].numVertices, &meshSeqVec[0].position0Vec[0]);
		myMesh.createVBO(4, meshSeqVec[0].numVertices, &meshSeqVec[0].colors0Vec[0]);
	}

	{
		MeshFile mfGrid;
		mfGrid.createGridWire(4.0f,4.0f,8,8);
		mfGrid.setColorVec(1.0f,1.0f,1.0f);
		mfGrid.flattenIndices();

		myMeshWorldGrid.create( mfGrid );
	}
	
	progAlbedoVertCol.addFile("albedo_vertcolor.vert", GL_VERTEX_SHADER);
	progAlbedoVertCol.addFile("albedo_vertcolor.frag", GL_FRAGMENT_SHADER);
	progAlbedoVertCol.buildProgram();
	
	// init gl state
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	LOG( LogLine() << "Use the arrow keys to go through the animation sequence!\n" ); 
}

void App::Update()
{
	float translationScale = 0.0035f;
	float rotationScale = 0.0015f;

	if( normalKeyState['w'] || normalKeyState['W'] )
	{
		myCamera.updateTranslationForwardBackward(-translationScale);
	}

	if( normalKeyState['s'] || normalKeyState['S'] )
	{
		myCamera.updateTranslationForwardBackward(translationScale);
	}

	if( normalKeyState['a'] || normalKeyState['A'] )
	{
		myCamera.updateTranslationLeftRight(-translationScale);
	}

	if( normalKeyState['d'] || normalKeyState['D'] )
	{
		myCamera.updateTranslationLeftRight(translationScale);
	}

	if( normalKeyState['q'] || normalKeyState['Q'] )
	{
		myCamera.updateTranslationUpDown(-translationScale);
	}

	if( normalKeyState['e'] || normalKeyState['E'] )
	{
		myCamera.updateTranslationUpDown(translationScale);
	}

	if( mouseState[GLUT_LEFT_BUTTON] )
	{
		if(mouseX_old != -1 && mouseY_old != -1)
		{
			myCamera.updateRotation(rotationScale * (mouseX - mouseX_old), rotationScale * (mouseY - mouseY_old), 0.0f);
		}

		mouseX_old = mouseX;
		mouseY_old = mouseY;
	}
	else
	{
		mouseX_old = -1,
		mouseY_old = -1;
	}

	if(specialKeyState[GLUT_KEY_LEFT])
	{
		--currID;
		currID = max(currID, 0);

		specialKeyState[GLUT_KEY_LEFT] = false;
	}

	if(specialKeyState[GLUT_KEY_RIGHT])
	{
		++currID;
		currID = min(currID, meshSeqSize-1);

		specialKeyState[GLUT_KEY_RIGHT] = false;
	}
}

void App::Render()
{
	myCamera.setProjParams( 60.0f, (float)(currWidth)/(float)(currHeight), 0.25f, 100.0f );
	myCamera.updateRays();

	glViewport(0, 0, currWidth, currHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// continuous update from host based on currID
	{
		myMesh.updateVBO(0, &meshSeqVec[currID].position0Vec[0]);
		myMesh.updateVBO(4, &meshSeqVec[currID].colors0Vec[0]);
	}

	{
		progAlbedoVertCol.useProgram();

		progAlbedoVertCol.setFloatMatrix44( myCamera.viewMat * myCamera.projMat, "u_MVPMat" );
		myMesh.render( progAlbedoVertCol.getAttribLocations() );

		progAlbedoVertCol.setFloatMatrix44(myCamera.viewMat * myCamera.projMat, "u_MVPMat");
		myMeshWorldGrid.render( progAlbedoVertCol.getAttribLocations() );

		glUseProgram(0);
	}
	
	CHECK_GL;
}

void App::Terminate()
{
	progAlbedoVertCol.clear();
	
	myMesh.clear();
	myMeshWorldGrid.clear();

	CHECK_GL;
}

std::string App::GetName()
{
	return std::string("gl_mesh_seq");
}