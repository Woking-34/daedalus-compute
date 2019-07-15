#include "App_flex_dambreak.h"

#include "system/log.h"
#include "system/timer.h"
#include "system/filesystem.h"

App AppInstance;

App::App()
{
	currWidth = 800;
	currHeight = 800;

	vboPos = 0;
	vboCol = 0;
}

App::~App()
{

}

void App::PrintCommandLineHelp()
{
	GLUTApplication::PrintCommandLineHelp();
}

void App::Initialize()
{
	InitFlexScene();

	{
		glGenBuffers(1, &vboPos);
		glBindBuffer(GL_ARRAY_BUFFER, vboPos);
		glBufferData(GL_ARRAY_BUFFER, initPos.size() * 4 * sizeof(float), initPos.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	{
		glGenBuffers(1, &vboCol);
		glBindBuffer(GL_ARRAY_BUFFER, vboCol);
		glBufferData(GL_ARRAY_BUFFER, initCol.size() * 4 * sizeof(float), initCol.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	{
		MeshFile mfAABB;
		//mfAABB.createBoxWire(g_sceneUpper.x - g_sceneLower.x, g_sceneUpper.y - g_sceneLower.y, g_sceneUpper.z - g_sceneLower.z);
		mfAABB.createBoxWire(g_sceneLower.x, g_sceneLower.y, g_sceneLower.z, g_sceneUpper.x, g_sceneUpper.y, g_sceneUpper.z);
		mfAABB.setColorVec(1.0f, 1.0f, 1.0f);
		mfAABB.flattenIndices();

		meshWorldAABB.create(mfAABB);
	}

	renderProgram_Particles.addFile("pointsprite3d.vert", GL_VERTEX_SHADER);
	renderProgram_Particles.addFile("pointsprite3d.frag", GL_FRAGMENT_SHADER);
	renderProgram_Particles.buildProgram();

	renderProgram_BBox.addFile("albedo_vertcolor.vert", GL_VERTEX_SHADER);
	renderProgram_BBox.addFile("albedo_vertcolor.frag", GL_FRAGMENT_SHADER);
	renderProgram_BBox.buildProgram();

	Vec3 g_camPos = Vec3((g_sceneLower.x + g_sceneUpper.x)*0.5f, g_sceneUpper.y * 0.85f, (g_sceneUpper.z - g_sceneLower.z) * 3.0f);

	const daedalus::Vec4f eye(g_camPos.x, g_camPos.y, g_camPos.z, 1.0f);
	const daedalus::Vec4f center(g_camPos.x, g_camPos.y - 0.5f, g_camPos.z - 3.0f, 1.0f);
	const daedalus::Vec4f up(0.0f, 1.0f, 0.0f, 0.0f);

	mainCamera.setViewParams(eye, center, up);
	mainCamera.setProjParams(60.0f, (float)(currWidth) / (float)(currHeight), 0.25f, 100.0f);
	mainCamera.updateRays();

	appCamera = &mainCamera;

	// init gl state
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);

	CHECK_GL;
}

void App::Update()
{

}

void App::Render()
{
	CHECK_GL;

	// number of active particles
	const int numParticles = NvFlexGetActiveCount(g_solver);

	// radius used for drawing
	float radius = Max(g_params.solidRestDistance, g_params.fluidRestDistance)*0.5f*g_pointScale;

	static int currIter = 0;

	// compute with flex
	{
		UpdateFlexFrame();
	}

	float fov = 60.0f;
	float aspect = float(currWidth) / float(currHeight);

	mainCamera.setProjParams(fov, aspect, 0.15f, 100.0f);
	mainCamera.updateRays();

	daedalus::Mat44f matVP = mainCamera.viewMat * mainCamera.projMat;

	glViewport(0, 0, currWidth, currHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	CHECK_GL;

	{
		renderProgram_BBox.useProgram();

		renderProgram_BBox.setFloatMatrix44(matVP, "u_MVPMat");
		meshWorldAABB.render(renderProgram_BBox.getAttribLocations());

		glUseProgram(0);
	}

#if !defined(HAVE_EGL)
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_PROGRAM_POINT_SIZE);
	//glEnable(GL_VERTEX_PROGRAM_POINT_SIZE); CHECK_GL
   
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
#endif
   
	renderProgram_Particles.useProgram();

	renderProgram_Particles.setFloatMatrix44(matVP, "u_MVMatrix");
	renderProgram_Particles.setFloatMatrix44(matVP, "u_MVPMatrix");

	renderProgram_Particles.setFloatValue(radius, "u_pSize");
	renderProgram_Particles.setFloatValue(currHeight / tanf(fov*0.5f*(float)M_PI/180.0f), "u_pScale");

	GLint positionHandleParticles = renderProgram_Particles.getAttribLocation("a_Position0");
	GLint colorHandleParticles = renderProgram_Particles.getAttribLocation("a_Color0");

	glBindBuffer(GL_ARRAY_BUFFER, vboPos);
	glVertexAttribPointer(positionHandleParticles, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glBindBuffer(GL_ARRAY_BUFFER, vboCol);
	glVertexAttribPointer(colorHandleParticles, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	
	glEnableVertexAttribArray(positionHandleParticles);  
	glEnableVertexAttribArray(colorHandleParticles);

	glDrawArrays(GL_POINTS, 0, numParticles);

	glDisableVertexAttribArray(positionHandleParticles);
	glDisableVertexAttribArray(colorHandleParticles);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glUseProgram(0);

#if !defined(HAVE_EGL)
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_PROGRAM_POINT_SIZE);
	//glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
#endif

	++currIter;
}

void App::Terminate()
{
	renderProgram_BBox.clear();
	renderProgram_Particles.clear();

	meshWorldAABB.clear();

	CHECK_GL;
}

std::string App::GetName()
{
	return std::string("flex_dambreak");
}