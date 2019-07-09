#include "App_gl_mesh_data.h"

#include "system/log.h"
#include "system/filesystem.h"

using namespace daedalus;

App AppInstance;

App::App() : numScenes(3)
{
	currWidth = 800;
	currHeight = 800;

	camCenter =  Vec4f(0.0f, 0.0f, 0.0f, 1.0f);
	camUp = Vec4f(0.0f, 1.0f, 0.0f, 0.0f);

	camFOV = 60.0f;
	camNear = 0.25f;
	camFar = 100.0f;

	camInitPos[0] = Vec4f(0.0f, 0.0f,-3.5f, 1.0f);
	camInitPos[1] = Vec4f(0.0f, 2.5f, 3.5f, 1.0f);
	camInitPos[2] = Vec4f(0.0f, 0.0f,-3.5f, 1.0f);

	memset(sceneSelcet, 0, sizeof(sceneSelcet));
	sceneSelcet[0] = 1;
	camEye = camInitPos[0];

	ballRad = 1.0f;
	clothSize = ballRad*2.0f * 5;

	memset(texIds, 0, sizeof(texIds)/sizeof(texIds[0]));
}

App::~App()
{

}

void App::Initialize()
{
	myCamera.setViewParams( camEye, camCenter, camUp );
	myCamera.setProjParams( camFOV, (float)(currWidth)/(float)(currHeight), camNear, camFar );
	myCamera.updateRays();

	{
		MeshFile myMeshFile0;
		myMeshFile0.setName("waterglass.obj", "assets/glass_objs/");
		myMeshFile0.loadFromFile();
		myMeshFile0.flattenIndices();
		
		MeshFile myMeshFile1;
		myMeshFile1.setName("wineglass.obj", "assets/glass_objs/");
		myMeshFile1.loadFromFile();
		myMeshFile1.flattenIndices();
		
		MeshFile myMeshFile2;
		myMeshFile2.setName("cognacglass.obj", "assets/glass_objs/");
		myMeshFile2.loadFromFile();
		myMeshFile2.flattenIndices();

		MeshFile myMeshFile;
		myMeshFile.setName("glass_scene.gen", "assets/glass_objs/");
		myMeshFile.add(myMeshFile0, createTranslationMatrix( 0.0f, 0.0f, 3.0f));
		myMeshFile.add(myMeshFile1, createTranslationMatrix( 3.0f, 0.0f,-3.0f));
		myMeshFile.add(myMeshFile2, createTranslationMatrix(-3.0f, 0.0f,-3.0f));
		myMeshFile.printInfo();
		
		myMeshFile.copyNormalsToColors();
		myMeshFile.scaleToUnitCube();

		myMeshFile0.copyNormalsToColors();
		myMeshFile0.scaleToUnitCube();

		myMesh.create( myMeshFile );
	}

	{
		MeshFile mfAABB;
		mfAABB.createBoxWire(2.0f,2.0f,2.0f);
		mfAABB.setColorVec(1.0f,1.0f,1.0f);
		mfAABB.flattenIndices();

		myMeshWorldAABB.create( mfAABB );
	}

	{
		MeshFile mfGrid;
		mfGrid.createGridWire(2.0f,2.0f,8,8);
		mfGrid.setColorVec(1.0f,1.0f,1.0f);
		mfGrid.flattenIndices();

		myMeshWorldGrid.create( mfGrid );
	}

	{
		MeshFile mfGrid;
		mfGrid.createGridTris(clothSize, clothSize, 10, 10);
		mfGrid.flattenIndices();

		meshGridPool.create( mfGrid );
	}

	{
		MeshFile mfSpeher;
		mfSpeher.createSphereTris(ballRad, 32, 32);
		mfSpeher.flattenIndices();

		meshSpherePool.create( mfSpeher );
	}

	{
		MeshFile mfCornell;
		createCornellBox(mfCornell);
		mfCornell.scaleToUnitCube();
		mfCornell.flattenIndices();

		meshCornell.create( mfCornell );
	}

	progAlbedoVertCol.addFile("albedo_vertcolor.vert", GL_VERTEX_SHADER);
	progAlbedoVertCol.addFile("albedo_vertcolor.frag", GL_FRAGMENT_SHADER);
	progAlbedoVertCol.buildProgram();

	progAlbedoTexCol.addFile("albedo_texcolor.vert", GL_VERTEX_SHADER);
	progAlbedoTexCol.addFile("albedo_texcolor.frag", GL_FRAGMENT_SHADER);
	progAlbedoTexCol.buildProgram();

	for(int i = 0; i < 15; ++i)
	{
		std::stringstream baseNameSS;
		baseNameSS << "pool/";
		baseNameSS << "pool_";
		baseNameSS << formatInt(i+1, 2, '0');
		baseNameSS << ".ppm";

		ImageData image;
		image.loadPBM( baseNameSS.str() );

		glGenTextures(1, texIds+i);
		glBindTexture(GL_TEXTURE_2D, texIds[i]);

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, image.getWidth(), image.getHeight(), 0, GL_RGB, GL_UNSIGNED_BYTE, image.getData());
		glGenerateMipmap(GL_TEXTURE_2D);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	{
		ImageData image;
		image.loadPBM( "pool/cloth.ppm" );

		glGenTextures(1, texIds+15);
		glBindTexture(GL_TEXTURE_2D, texIds[15]);

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, image.getWidth(), image.getHeight(), 0, GL_RGB, GL_UNSIGNED_BYTE, image.getData());
		glGenerateMipmap(GL_TEXTURE_2D);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	unsigned ballIndex = 0;
	for(unsigned i=0; i<5; ++i)
	{
		for(unsigned j=0; j<=i; ++j)
		{
			// pool triangle
			Vec4f center;
			center.x = -2.0f*(float)j+(float)i;
			center.y = -1.0f;
			center.z = -2.0f*(float)i;
			center.w = 1.0f;

			// random rotate
			Mat44f rotMatX = createRotationX( deg2rad(180.0f * (2.0f * randomFLT() - 1.0f)) );
			Mat44f rotMatY = createRotationY( deg2rad(180.0f * (2.0f * randomFLT() - 1.0f)) );
			Mat44f rotMatZ = createRotationZ( deg2rad(180.0f * (2.0f * randomFLT() - 1.0f)) );

			Mat44f rotMat = rotMatZ * rotMatY * rotMatX;
			//rotMat = Mat44f(one);

			ballMatrices[ballIndex] = rotMat * createTranslationMatrix(center);

			++ballIndex;
		}
	}

	// init gl state
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	LOG( LogLine() << "Press F1-F3 to switch between different scenes!\n" ); 
}

void App::Update()
{
	if(specialKeyState[GLUT_KEY_F1])
	{
		specialKeyState[GLUT_KEY_F1] = false;

		memset(sceneSelcet, 0, sizeof(sceneSelcet));
		sceneSelcet[0] = 1;

		myCamera.setViewParams( camInitPos[0], camCenter, camUp );
	}

	if(specialKeyState[GLUT_KEY_F2])
	{
		specialKeyState[GLUT_KEY_F2] = false;

		memset(sceneSelcet, 0, sizeof(sceneSelcet));
		sceneSelcet[1] = 1;

		myCamera.setViewParams( camInitPos[1], camCenter, camUp );
	}

	if(specialKeyState[GLUT_KEY_F3])
	{
		specialKeyState[GLUT_KEY_F3] = false;

		memset(sceneSelcet, 0, sizeof(sceneSelcet));
		sceneSelcet[2] = 1;

		myCamera.setViewParams( camInitPos[2], camCenter, camUp );
	}

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
}

void App::Render()
{
	myCamera.setProjParams( 60.0f, (float)(currWidth)/(float)(currHeight), 0.25f, 100.0f );
	myCamera.updateRays();

	glViewport(0, 0, currWidth, currHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if(sceneSelcet[0])
	{
		progAlbedoVertCol.useProgram();

		progAlbedoVertCol.setFloatMatrix44( myCamera.viewMat * myCamera.projMat, "u_MVPMat" );
		myMesh.render( progAlbedoVertCol.getAttribLocations() );

		progAlbedoVertCol.setFloatMatrix44(myCamera.viewMat * myCamera.projMat, "u_MVPMat");
		myMeshWorldAABB.render( progAlbedoVertCol.getAttribLocations() );

		progAlbedoVertCol.setFloatMatrix44(createTranslationMatrix(0.0f, -1.0f, 0.0f) * myCamera.viewMat * myCamera.projMat, "u_MVPMat");
		myMeshWorldGrid.render( progAlbedoVertCol.getAttribLocations() );

		glUseProgram(0);
	}
	else if(sceneSelcet[1])
	{
		progAlbedoTexCol.useProgram();

		progAlbedoTexCol.setIntValue(0, "albedoTex");

		for(int i = 0; i < 15; ++i)
		{
			glBindTexture(GL_TEXTURE_2D, texIds[i]);
			progAlbedoTexCol.setFloatMatrix44(ballMatrices[i] * myCamera.viewMat * myCamera.projMat, "u_MVPMat");
			meshSpherePool.render( progAlbedoTexCol.getAttribLocations() );
		}
	
		glBindTexture(GL_TEXTURE_2D, texIds[15]);
		progAlbedoTexCol.setFloatMatrix44(createTranslationMatrix(0.0f, -2.0f, -clothSize / 2.0f + ballRad) * myCamera.viewMat * myCamera.projMat, "u_MVPMat");
		meshGridPool.render( progAlbedoTexCol.getAttribLocations() );

		glUseProgram(0);
	}
	else if(sceneSelcet[2])
	{
		progAlbedoVertCol.useProgram();

		progAlbedoVertCol.setFloatMatrix44(myCamera.viewMat * myCamera.projMat, "u_MVPMat");
		meshCornell.render( progAlbedoVertCol.getAttribLocations() );

		glUseProgram(0);
	}

	CHECK_GL;
}

void App::Terminate()
{
	progAlbedoVertCol.clear();
	progAlbedoTexCol.clear();

	myMesh.clear();
	myMeshWorldAABB.clear();
	myMeshWorldGrid.clear();

	meshSpherePool.clear();
	meshGridPool.clear();

	meshCornell.clear();

	glDeleteTextures(sizeof(texIds)/sizeof(texIds[0]), texIds);

	CHECK_GL;
}

std::string App::GetName()
{
	return std::string("gl_mesh_data");
}

void App::createCornellBox(MeshFile& mfCornell)
{
	const Vec4f white = Vec4f( 0.80f, 0.80f, 0.80f, 1.0f );
	const Vec4f green = Vec4f( 0.05f, 0.80f, 0.05f, 1.0f );
	const Vec4f red   = Vec4f( 0.80f, 0.05f, 0.05f, 1.0f );

	// Floor
	mfCornell.addParallelogram(
		Vec4f( 0.0f, 0.0f, 0.0f, 1.0f ),
		Vec4f( 0.0f, 0.0f, 559.2f, 0.0f ),
		Vec4f( 556.0f, 0.0f, 0.0f, 0.0f ),
		white
		);

	// Ceiling
	mfCornell.addParallelogram(
		Vec4f( 0.0f, 548.8f, 0.0f, 1.0f ),
		Vec4f( 556.0f, 0.0f, 0.0f, 0.0f ),
		Vec4f( 0.0f, 0.0f, 559.2f, 0.0f ),
		white
		);

	// Back wall
	mfCornell.addParallelogram(
		Vec4f( 0.0f, 0.0f, 559.2f, 1.0f),
		Vec4f( 0.0f, 548.8f, 0.0f, 0.0f),
		Vec4f( 556.0f, 0.0f, 0.0f, 0.0f),
		white
		);

	// Right wall
	mfCornell.addParallelogram(
		Vec4f( 0.0f, 0.0f, 0.0f, 1.0f ),
		Vec4f( 0.0f, 548.8f, 0.0f, 0.0f ),
		Vec4f( 0.0f, 0.0f, 559.2f, 0.0f ),
		green
		);

	// Left wall
	mfCornell.addParallelogram(
		Vec4f( 556.0f, 0.0f, 0.0f, 1.0f ),
		Vec4f( 0.0f, 0.0f, 559.2f, 0.0f ),
		Vec4f( 0.0f, 548.8f, 0.0f, 0.0f ),
		red
		);

	// Short block
	mfCornell.addParallelogram(
		Vec4f( 130.0f, 165.0f, 65.0f, 1.0f),
		Vec4f( -48.0f, 0.0f, 160.0f, 0.0f),
		Vec4f( 160.0f, 0.0f, 49.0f, 0.0f),
		white
		);

	mfCornell.addParallelogram(
		Vec4f( 290.0f, 0.0f, 114.0f, 1.0f),
		Vec4f( 0.0f, 165.0f, 0.0f, 0.0f),
		Vec4f( -50.0f, 0.0f, 158.0f, 0.0f),
		white
		);
	mfCornell.addParallelogram(
		Vec4f( 130.0f, 0.0f, 65.0f, 1.0f),
		Vec4f( 0.0f, 165.0f, 0.0f, 0.0f),
		Vec4f( 160.0f, 0.0f, 49.0f, 0.0f),
		white);
	mfCornell.addParallelogram(
		Vec4f( 82.0f, 0.0f, 225.0f, 1.0f),
		Vec4f( 0.0f, 165.0f, 0.0f, 0.0f),
		Vec4f( 48.0f, 0.0f, -160.0f, 0.0f),
		white
		);
	mfCornell.addParallelogram(
		Vec4f( 240.0f, 0.0f, 272.0f, 1.0f),
		Vec4f( 0.0f, 165.0f, 0.0f, 0.0f),
		Vec4f( -158.0f, 0.0f, -47.0f, 0.0f),
		white
		);

	// Tall block
	mfCornell.addParallelogram(
		Vec4f( 423.0f, 330.0f, 247.0f, 1.0f),
		Vec4f( -158.0f, 0.0f, 49.0f, 0.0f),
		Vec4f( 49.0f, 0.0f, 159.0f, 0.0f),
		white
		);
	mfCornell.addParallelogram(
		Vec4f( 423.0f, 0.0f, 247.0f, 1.0f),
		Vec4f( 0.0f, 330.0f, 0.0f, 0.0f),
		Vec4f( 49.0f, 0.0f, 159.0f, 0.0f),
		white);
	mfCornell.addParallelogram(
		Vec4f( 472.0f, 0.0f, 406.0f, 1.0f),
		Vec4f( 0.0f, 330.0f, 0.0f, 0.0f),
		Vec4f( -158.0f, 0.0f, 50.0f, 0.0f),
		white
		);
	mfCornell.addParallelogram(
		Vec4f( 314.0f, 0.0f, 456.0f, 1.0f),
		Vec4f( 0.0f, 330.0f, 0.0f, 0.0f),
		Vec4f( -49.0f, 0.0f, -160.0f, 0.0f),
		white
		);
	mfCornell.addParallelogram(
		Vec4f( 265.0f, 0.0f, 296.0f, 1.0f),
		Vec4f( 0.0f, 330.0f, 0.0f, 0.0f),
		Vec4f( 158.0f, 0.0f, -49.0f, 0.0f),
		white
		);

	mfCornell.tesselate();
	mfCornell.tesselate();
	mfCornell.tesselate();
	mfCornell.tesselate();
}