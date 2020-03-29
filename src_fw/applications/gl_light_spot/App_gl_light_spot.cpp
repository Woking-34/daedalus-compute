#include "App_gl_light_spot.h"

#include "system/log.h"
#include "system/filesystem.h"

using namespace daedalus;

App AppInstance;

App::App() : texIdCrate(0), texIdChecker(0)
{
	currWidth = 800;
	currHeight = 800;

	isSpotInput = false;

	range1 = 17.5f;
	range2 = 22.5f;

	angle1 = 20.0f;
	angle2 = 25.0f;

	posSource_SpotLight = Vec4f(0.0f, 15.0f, 0.0f, 1.0f);
	posTarget_SpotLight = Vec4f(0.0f, 0.0f, 0.0f, 1.0f);

	shadowSizeX = 512;
	shadowSizeY = 512;

	renderMode = 2;
}

App::~App()
{

}

void App::Initialize()
{
	const Vec4f eye(0.0f, 10.0f, 10.0f, 1.0f);
	const Vec4f center(0.0f, 0.0f, 0.0f, 1.0f);
	const Vec4f up(0.0f, 1.0f, 0.0f, 0.0f);

	mainCamera.setViewParams( eye, center, up );
	mainCamera.setProjParams( 60.0f, (float)(currWidth)/(float)(currHeight), 0.25f, 100.0f );
	mainCamera.updateRays();

	appCamera = &mainCamera;

	progAlbedo.addFile("albedo_meshcolor.vert", GL_VERTEX_SHADER);
	progAlbedo.addFile("albedo_meshcolor.frag", GL_FRAGMENT_SHADER);
	progAlbedo.buildProgram();

	progAlbedoTex.addFile("albedo_texcolor.vert", GL_VERTEX_SHADER);
	progAlbedoTex.addFile("albedo_texcolor.frag", GL_FRAGMENT_SHADER);
	progAlbedoTex.buildProgram();

	progLightSpot.addFile("lightspot_texcolor.vert", GL_VERTEX_SHADER);
	progLightSpot.addFile("lightspot_texcolor.frag", GL_FRAGMENT_SHADER);
	progLightSpot.buildProgram();

	progLightSpotShadow.addFile("lightspotsh_texcolor.vert", GL_VERTEX_SHADER);
	progLightSpotShadow.addFile("lightspotsh_texcolor.frag", GL_FRAGMENT_SHADER);
	progLightSpotShadow.buildProgram();

	MeshFile mfGrid;
	mfGrid.createGridTris(15.0f, 15.0f, 8, 8);
	mfGrid.flattenIndices();
	meshGrid.create( mfGrid );

	MeshFile mfBox;
	mfBox.createBoxTris(2.0f, 2.0f, 2.0f);
	mfBox.flattenIndices();
	meshBox.create( mfBox );

	MeshFile mfSphere;
	mfSphere.createSphereTris(1.0f, 32, 32);
	mfSphere.flattenIndices();
	meshSphere.create( mfSphere );

	MeshFile mfCone;
	mfCone.createConeWire(range2, angle2, 32);
	mfCone.flattenIndices();
	meshConeWire.create( mfCone );

	fboShadow.createTex2D(shadowSizeX, shadowSizeY, true);

	{
		ImageData image;
		image.loadBMP( "crate.bmp" );

		glGenTextures(1, &texIdCrate);
		glBindTexture(GL_TEXTURE_2D, texIdCrate);

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, image.getWidth(), image.getHeight(), 0, GL_RGB, GL_UNSIGNED_BYTE, image.getData());
		glGenerateMipmap(GL_TEXTURE_2D);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	{
		ImageData image;
		image.loadBMP( "checker.bmp" );

		glGenTextures(1, &texIdChecker);
		glBindTexture(GL_TEXTURE_2D, texIdChecker);

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, image.getWidth(), image.getHeight(), 0, GL_RGB, GL_UNSIGNED_BYTE, image.getData());
		glGenerateMipmap(GL_TEXTURE_2D);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	// init gl state
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	CHECK_GL;

	LOG( LogLine() << "Press F1/F2/F3 to switch between different rendering modes!\n" );
	LOG( LogLine() << "Press space to switch between camera/light control!\n" );
}

void App::Update()
{
	if(specialKeyState[GLUT_KEY_F1])
	{
		specialKeyState[GLUT_KEY_F1] = false;
		renderMode = 0;
	}

	if(specialKeyState[GLUT_KEY_F2])
	{
		specialKeyState[GLUT_KEY_F2] = false;
		renderMode = 1;
	}

	if(specialKeyState[GLUT_KEY_F3])
	{
		specialKeyState[GLUT_KEY_F3] = false;
		renderMode = 2;
	}

	if(normalKeyState[' '] == true)
	{
		normalKeyState[' '] = false;
		isSpotInput = !isSpotInput;
	}

	float translationScale = 0.0035f * delta_time;
	float rotationScale = 0.0015f;

	if(isSHIFTPressed)
	{
		translationScale *= 10.0f;
	}

	if(isSpotInput == false)
	{
		appCamera = &mainCamera;
	}
	else
	{
		appCamera = nullptr;

		if( normalKeyState['w'] || normalKeyState['W'] )
		{
			posSource_SpotLight -= Vec4f(0.0f, 0.0f, translationScale, 0.0f);
		}

		if( normalKeyState['s'] || normalKeyState['S'] )
		{
			posSource_SpotLight += Vec4f(0.0f, 0.0f, translationScale, 0.0f);
		}

		if( normalKeyState['a'] || normalKeyState['A'] )
		{
			posSource_SpotLight -= Vec4f(translationScale, 0.0f, 0.0f, 0.0f);
		}

		if( normalKeyState['d'] || normalKeyState['D'] )
		{
			posSource_SpotLight += Vec4f(translationScale, 0.0f, 0.0f, 0.0f);
		}

		if( normalKeyState['q'] || normalKeyState['Q'] )
		{
			posSource_SpotLight -= Vec4f(0.0f, translationScale, 0.0f, 0.0f);
		}

		if( normalKeyState['e'] || normalKeyState['E'] )
		{
			posSource_SpotLight += Vec4f(0.0f, translationScale, 0.0f, 0.0f);
		}
	}
}

void App::Render()
{
	mainCamera.setProjParams( 60.0f, (float)(currWidth)/(float)(currHeight), 0.25f, 100.0f );
	mainCamera.updateRays();

	glViewport(0, 0, currWidth, currHeight);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	{
		if(renderMode == 0)
			RenderAlbedo(mainCamera.viewMat, mainCamera.projMat);
		else if(renderMode == 1)
			RenderShaded();
		else if(renderMode == 2)
			RenderShadedShadow();
	}

	{
		RenderDebug();
	}

	CHECK_GL;
}

void App::Terminate()
{
	progAlbedo.clear();
	progAlbedoTex.clear();

	progLightSpot.clear();
	progLightSpotShadow.clear();
	
	meshGrid.clear();
	meshBox.clear();
	meshSphere.clear();

	meshConeWire.clear();

	if(texIdCrate)
	{
		glDeleteTextures(1, &texIdCrate);
		texIdCrate = 0;
	}

	if(texIdChecker)
	{
		glDeleteTextures(1, &texIdChecker);
		texIdChecker = 0;
	}

	CHECK_GL;
}

std::string App::GetName()
{
	return std::string("gl_light_spot");
}

void App::RenderDebug()
{
	Vec4f dirInit_SpotLight = Vec4f(0.0, -1.0f, 0.0f, 0.0f);	
	Vec4f dir_SpotLight = posTarget_SpotLight - posSource_SpotLight;

	Vec4f rotAxis = cross(dirInit_SpotLight, dir_SpotLight);
	float theta = 0.0f;

	if(rotAxis.x == 0.0f && rotAxis.y == 0.0f && rotAxis.z == 0.0f)
	{
		rotAxis = dirInit_SpotLight;
		theta = 0.0f;
	}
	else
	{
		dir_SpotLight = normalize(dir_SpotLight);
		rotAxis = normalize(rotAxis);

		theta = dot(dir_SpotLight, dirInit_SpotLight);
		theta = acos(theta);
	}

	progAlbedo.useProgram();

	Mat44f modelMat = createRotationMatrix(rotAxis, -theta) * createTranslationMatrix(posSource_SpotLight);
	progAlbedo.setFloatMatrix44( modelMat * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );
	progAlbedo.setFloatVector4( Vec4f(0.0f, 1.0f, 0.0f, 1.0f), "u_Color");
	meshConeWire.render( progAlbedo.getAttribLocations() );

	glUseProgram(0);
}

void App::RenderAlbedo(const Mat44f& viewMat, const Mat44f& projMat)
{
	progAlbedoTex.useProgram();

	progAlbedoTex.setIntValue(0, "albedoTex");
	glActiveTexture(GL_TEXTURE0 + 0);

	{
		glBindTexture(GL_TEXTURE_2D, texIdChecker);

		Mat44f modelMat = Mat44f(one);
		progAlbedoTex.setFloatMatrix44( modelMat * viewMat * projMat, "u_MVPMat" );
		meshGrid.render( progAlbedoTex.getAttribLocations() );
	}

	{
		glBindTexture(GL_TEXTURE_2D, texIdChecker);

		Mat44f modelMat = createTranslationMatrix(0.0f, 2.5f, -2.5f) * createRotationY(curr_time * 0.001f);
		progAlbedoTex.setFloatMatrix44( modelMat * viewMat * projMat, "u_MVPMat" );
		meshSphere.render( progAlbedoTex.getAttribLocations() );
	}

	{
		glBindTexture(GL_TEXTURE_2D, texIdCrate);

		Mat44f modelMat = createTranslationMatrix(-5.0f, 1.0f, 0.0f);
		progAlbedoTex.setFloatMatrix44( modelMat * viewMat * projMat, "u_MVPMat" );
		meshBox.render( progAlbedoTex.getAttribLocations() );
	}

	{
		glBindTexture(GL_TEXTURE_2D, texIdCrate);

		Mat44f modelMat = createTranslationMatrix(5.0f, 1.0f, 0.0f);
		progAlbedoTex.setFloatMatrix44( modelMat * viewMat * projMat, "u_MVPMat" );
		meshBox.render( progAlbedoTex.getAttribLocations() );
	}

	glUseProgram(0);
}

void App::RenderShaded()
{
	progLightSpot.useProgram();

	progLightSpot.setIntValue(0, "albedoTex");
	glActiveTexture(GL_TEXTURE0 + 0);	
	
	progLightSpot.setFloatVector4(mainCamera.getPosition(), "eyePosW");
	progLightSpot.setFloatVector4(posSource_SpotLight, "lightPosW");
	progLightSpot.setFloatVector4(posTarget_SpotLight - posSource_SpotLight, "lightDirW");

	progLightSpot.setFloatVector4(Vec4f(1.0f, 1.0f, 1.0f, 1.0f), "diffMaterial");
	progLightSpot.setFloatVector4(Vec4f(1.0f, 1.0f, 1.0f, 10.0f), "specMaterial");
	progLightSpot.setFloatVector4(Vec4f(0.25f, 0.25f, 0.25f, 1.0f), "ambMaterial");
	progLightSpot.setFloatVector4(Vec4f(range1, range2, deg2rad(angle1), deg2rad(angle2)), "attenLight");

	{
		glBindTexture(GL_TEXTURE_2D, texIdChecker);

		Mat44f modelMat = Mat44f(one);
		progLightSpot.setFloatMatrix44( modelMat * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );
		progLightSpot.setFloatMatrix44( modelMat, "u_MMat" );
		meshGrid.render( progLightSpot.getAttribLocations() );
	}

	{
		glBindTexture(GL_TEXTURE_2D, texIdChecker);

		Mat44f modelMat = createTranslationMatrix(0.0f, 2.5f, -2.5f) * createRotationY(curr_time * 0.001f);
		progLightSpot.setFloatMatrix44( modelMat * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );
		progLightSpot.setFloatMatrix44( modelMat, "u_MMat" );
		meshSphere.render( progLightSpot.getAttribLocations() );
	}

	{
		glBindTexture(GL_TEXTURE_2D, texIdCrate);

		Mat44f modelMat = createTranslationMatrix(-5.0f, 1.0f, 0.0f);
		progLightSpot.setFloatMatrix44( modelMat * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );
		progLightSpot.setFloatMatrix44( modelMat, "u_MMat" );
		meshBox.render( progLightSpot.getAttribLocations() );
	}

	{
		glBindTexture(GL_TEXTURE_2D, texIdCrate);

		Mat44f modelMat = createTranslationMatrix(5.0f, 1.0f, 0.0f);
		progLightSpot.setFloatMatrix44( modelMat * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );
		progLightSpot.setFloatMatrix44( modelMat, "u_MMat" );
		meshBox.render( progLightSpot.getAttribLocations() );
	}

	glUseProgram(0);
}

void App::RenderShadedShadow()
{
	Mat44f spotViewMat;
	Mat44f spotProjMat;

	// generate matrices for spotlight-shadow
	{
		spotViewMat = createViewLookAtMatrix(posSource_SpotLight, posTarget_SpotLight, Vec4f(1.0f, 0.0f, 0.0f, 0.0f));
		spotProjMat = createProjectionPerspectiveMatrix(angle2 * 2.0f, 1.0f, 0.25f, range2);
	}

	{
		fboShadow.bind();

		glViewport(0, 0, fboShadow.getWidth(), fboShadow.getHeight());
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		RenderAlbedo(spotViewMat, spotProjMat);

		fboShadow.unbind();
	}

	{
		glViewport(0, 0, currWidth, currHeight);

		progLightSpotShadow.useProgram();

		progLightSpotShadow.setIntValue(1, "shadowMap");
		glActiveTexture(GL_TEXTURE0 + 1);
		glBindTexture(GL_TEXTURE_2D, fboShadow.getTexID());

		progLightSpotShadow.setFloatVector4(mainCamera.getPosition(), "eyePosW");
		progLightSpotShadow.setFloatVector4(posSource_SpotLight, "lightPosW");
		progLightSpotShadow.setFloatVector4(posTarget_SpotLight - posSource_SpotLight, "lightDirW");

		progLightSpotShadow.setFloatVector4(Vec4f(1.0f, 1.0f, 1.0f, 1.0f), "diffMaterial");
		progLightSpotShadow.setFloatVector4(Vec4f(1.0f, 1.0f, 1.0f, 10.0f), "specMaterial");
		progLightSpotShadow.setFloatVector4(Vec4f(0.25f, 0.25f, 0.25f, 1.0f), "ambMaterial");
		progLightSpotShadow.setFloatVector4(Vec4f(range1, range2, deg2rad(angle1), deg2rad(angle2)), "attenLight");

		{
			progLightSpotShadow.setIntValue(0, "albedoTex");
			glActiveTexture(GL_TEXTURE0 + 0);	
			glBindTexture(GL_TEXTURE_2D, texIdChecker);

			Mat44f modelMat = Mat44f(one);
			progLightSpotShadow.setFloatMatrix44( modelMat * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );
			progLightSpotShadow.setFloatMatrix44( modelMat, "u_MMat" );
			progLightSpotShadow.setFloatMatrix44( modelMat * spotViewMat * spotProjMat, "u_LMat" );
			meshGrid.render( progLightSpotShadow.getAttribLocations() );
		}

		{
			progLightSpotShadow.setIntValue(0, "albedoTex");
			glActiveTexture(GL_TEXTURE0 + 0);	
			glBindTexture(GL_TEXTURE_2D, texIdChecker);

			Mat44f modelMat = createTranslationMatrix(0.0f, 2.5f, -2.5f) * createRotationY(curr_time * 0.001f);
			progLightSpotShadow.setFloatMatrix44( modelMat * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );
			progLightSpotShadow.setFloatMatrix44( modelMat, "u_MMat" );
			progLightSpotShadow.setFloatMatrix44( modelMat * spotViewMat * spotProjMat, "u_LMat" );
			meshSphere.render( progLightSpotShadow.getAttribLocations() );
		}

		{
			progLightSpotShadow.setIntValue(0, "albedoTex");
			glActiveTexture(GL_TEXTURE0 + 0);	
			glBindTexture(GL_TEXTURE_2D, texIdCrate);

			Mat44f modelMat = createTranslationMatrix(-5.0f, 1.0f, 0.0f);
			progLightSpotShadow.setFloatMatrix44( modelMat * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );
			progLightSpotShadow.setFloatMatrix44( modelMat, "u_MMat" );
			progLightSpotShadow.setFloatMatrix44( modelMat * spotViewMat * spotProjMat, "u_LMat" );
			meshBox.render( progLightSpotShadow.getAttribLocations() );
		}

		{
			progLightSpotShadow.setIntValue(0, "albedoTex");
			glActiveTexture(GL_TEXTURE0 + 0);	
			glBindTexture(GL_TEXTURE_2D, texIdCrate);

			Mat44f modelMat = createTranslationMatrix(5.0f, 1.0f, 0.0f);
			progLightSpotShadow.setFloatMatrix44( modelMat * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );
			progLightSpotShadow.setFloatMatrix44( modelMat, "u_MMat" );
			progLightSpotShadow.setFloatMatrix44( modelMat * spotViewMat * spotProjMat, "u_LMat" );
			meshBox.render( progLightSpotShadow.getAttribLocations() );
		}

		glUseProgram(0);
	}
}