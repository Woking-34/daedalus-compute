#include "App_comp_particles.h"

#ifdef COMPGL
	#include "COMPGL_PARTICLES.h"
#elif COMPCL
	#include "COMPCL_PARTICLES.h"
#elif COMPCU
	#include "COMPCU_PARTICLES.h"
#endif

#include "system/log.h"
#include "system/timer.h"
#include "system/filesystem.h"

App AppInstance;

App::App()
{
	useInterop = true;

	useCLGDevice = true;
	useCLPId = -1;
	useCLDId = -1;

	calculator = NULL;

	currWidth = 800;
	currHeight = 800;

	vboPos = 0;
	vboCol = 0;
	
	{
		wgsX = 64;
		wgsY = 1;
		wgsZ = 1;

		numParticles=4096;
		numParticlesDimX=16;
		numParticlesDimY=16;
		numParticlesDimZ=16;

		numParticles=32768;
		numParticlesDimX=32;
		numParticlesDimY=32;
		numParticlesDimZ=32;

		particleRadius = (0.5f / numParticlesDimX) * 0.5f;

		gridCells = (int)(0.5f / particleRadius);
		numGridCells = gridCells*gridCells*gridCells;
		numGridCellsPaddded = roundUp(numGridCells, wgsX);
	}
}

App::~App()
{

}

void App::PrintCommandLineHelp()
{
	GLUTApplication::PrintCommandLineHelp();

	LOG( "Options" );

#ifdef COMPCL
	LOG( "\t--ei useCLP = select OpenCL platform with index" );
	LOG( "\t--ei useCLD = select OpenCL device with index" );
	LOG( "\t--ez useCLGL = select OpenCL device for current OpenGL context" );
	LOG( "" );
#endif
}

void App::Initialize()
{
	ReadCmdArg(STRING(useInterop), useInterop);

	ReadCmdArg("useCLGL", useCLGDevice);
	ReadCmdArg("useCLP", useCLPId);
	ReadCmdArg("useCLD", useCLDId);

	ReadCmdArg(STRING(wgsX), wgsX);
	ReadCmdArg(STRING(wgsY), wgsY);
	ReadCmdArg(STRING(wgsZ), wgsZ);

	initPos = new Vec4f[numParticles];
	initVel = new Vec4f[numParticles];
	initCol = new Vec4f[numParticles];

	for(int k = 0; k < numParticlesDimZ; ++k)
	{
		for(int j = 0; j < numParticlesDimY; ++j)
		{
			for(int i = 0; i < numParticlesDimX; ++i)
			{
				initPos[ (k*numParticlesDimX*numParticlesDimY) + (j*numParticlesDimX) + i ] = 
					Vec4f(
						0.25f + particleRadius + i*2.0f*particleRadius + 0.15f * particleRadius * (0.5f - (float)rand()/RAND_MAX),
						0.05f + particleRadius + j*2.0f*particleRadius + 0.15f * particleRadius * (0.5f - (float)rand()/RAND_MAX),
						0.25f + particleRadius + k*2.0f*particleRadius + 0.15f * particleRadius * (0.5f - (float)rand()/RAND_MAX),
						1.0f
					);

				initCol[ (k*numParticlesDimX*numParticlesDimY) + (j*numParticlesDimX) + i ] = 
					Vec4f( (float)i / numParticlesDimX, (float)j / numParticlesDimY, (float)k / numParticlesDimZ, 1.0f );
			}
		}
	}

	for(int i = 0; i < numParticles; ++i)
	{
		initVel[i] = Vec4f( 0.0f, 0.0f, 0.0f, 0.0f );
	}

	{
		glGenBuffers(1, &vboPos);
		glBindBuffer(GL_ARRAY_BUFFER, vboPos);
		glBufferData(GL_ARRAY_BUFFER, numParticles*4*sizeof(float), initPos, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	{
		glGenBuffers(1, &vboCol);
		glBindBuffer(GL_ARRAY_BUFFER, vboCol);
		glBufferData(GL_ARRAY_BUFFER, numParticles*4*sizeof(float), initCol, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	{
		float lines2VerticesData[] =
		{
			// X, Y, Z, W
			// R, G, B, A
			0.0f, 0.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			1.0f, 0.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			1.0f, 0.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			1.0f, 0.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			1.0f, 0.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			0.0f, 0.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			0.0f, 0.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			0.0f, 0.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			// --- //

			0.0f, 1.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			1.0f, 1.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			1.0f, 1.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			1.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			1.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			0.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			0.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			0.0f, 1.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			// --- //

			0.0f, 0.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			0.0f, 1.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			1.0f, 0.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			1.0f, 1.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			1.0f, 0.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			1.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			0.0f, 0.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f,

			0.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f
		};

		{
			{
				glGenBuffers(1, &vboBBox);
				glBindBuffer(GL_ARRAY_BUFFER, vboBBox);
				glBufferData(GL_ARRAY_BUFFER, 3 * 8 * 2 * 4 * sizeof(float), lines2VerticesData, GL_STATIC_DRAW);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}
		}
	}

	renderProgram_Particles.addFile("pointsprite3d.vert", GL_VERTEX_SHADER);
	renderProgram_Particles.addFile("pointsprite3d.frag", GL_FRAGMENT_SHADER);
	renderProgram_Particles.buildProgram();

	renderProgram_BBox.addFile("albedo_vertcolor.vert", GL_VERTEX_SHADER);
	renderProgram_BBox.addFile("albedo_vertcolor.frag", GL_FRAGMENT_SHADER);
	renderProgram_BBox.buildProgram();

	const Vec4f eye(0.0f, 1.15f, 1.75f, 1.0f);
	const Vec4f center(0.0f, 0.0f, -1.25f, 1.0f);
	const Vec4f up(0.0f, 1.0f, 0.0f, 0.0f);

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

#ifdef COMPGL
	api = "GL";
	calculator = new COMPGL_PARTICLES;
#elif COMPCL
	api = "CL";
	calculator = new COMPCL_PARTICLES;
#elif COMPCU
	api = "CU";
	calculator = new COMPCU_PARTICLES;
#endif

	calculator->wgsX = wgsX;
	calculator->wgsY = wgsY;
	calculator->wgsZ = wgsZ;

	calculator->useInterop = useInterop;
	calculator->bufferPos = (float*)initPos;
	calculator->bufferCol = (float*)initCol;
	calculator->bufferVel = (float*)initVel;
	calculator->vboPos = vboPos;
	calculator->vboCol = vboCol;

	calculator->numParticles = numParticles;
	calculator->numParticlesDimX = numParticlesDimX;
	calculator->numParticlesDimY = numParticlesDimY;
	calculator->numParticlesDimZ = numParticlesDimZ;
	calculator->particleRadius = particleRadius;
	calculator->gridCells = gridCells;
	calculator->numGridCells = numGridCells;
	calculator->numGridCellsPaddded = numGridCellsPaddded;

	calculator->useCLGDevice = useCLGDevice;
	calculator->useCLPId = useCLPId;
	calculator->useCLDId = useCLDId;

	calculator->init();

	// refresh interop status based on device override
	useInterop = calculator->useInterop;

	LOG( LogLine() << "GLOBAL_WORK_SIZE: " << numParticles << "\n" );
	LOG( LogLine() << "LOCAL_WORK_SIZE: " << calculator->wgsX << "\n" );
	LOG( LogLine() << "INTEROP: " << (useInterop ? "ENABLED" : "DISABLED") << "\n" );
}

void App::Update()
{
}

void App::Render()
{
	CHECK_GL;

	static int currIter = 0;

	{
		// compute
		{
			calculator->compute();
		}

		// download
		if(useInterop == false)
		{
			calculator->download();
		}

		// upload
		if(useInterop == false)
		{
			{
				glBindBuffer(GL_ARRAY_BUFFER, calculator->vboPos);
				void* hostPtr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
				memcpy(hostPtr, calculator->bufferPos, numParticles * 4 * sizeof(float));
				glUnmapBuffer(GL_ARRAY_BUFFER);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}
		}
	}

	float fov = 60.0f;
	float aspect = float(currWidth) / float(currHeight);

	mainCamera.setProjParams(fov, aspect, 0.15f, 100.0f);
	mainCamera.updateRays();

	Mat44f matModel = createTranslationMatrix(-0.5f,0.0f,-0.5f);
	Mat44f matMV = matModel * mainCamera.viewMat;
	Mat44f matMVP = matModel * mainCamera.viewMat * mainCamera.projMat;

	glViewport(0, 0, currWidth, currHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	CHECK_GL;

	
#if !defined(HAVE_EGL)
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_PROGRAM_POINT_SIZE);
	//glEnable(GL_VERTEX_PROGRAM_POINT_SIZE); CHECK_GL
   
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
#endif
   
	renderProgram_Particles.useProgram();

	renderProgram_Particles.setFloatMatrix44(matMV, "u_MVMatrix");
	renderProgram_Particles.setFloatMatrix44(matMVP, "u_MVPMatrix");

	renderProgram_Particles.setFloatValue(particleRadius, "u_pSize");
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

	{
		renderProgram_BBox.useProgram();

		renderProgram_BBox.setFloatMatrix44(matMVP, "u_MVPMat");

		GLint positionHandleBBox = renderProgram_BBox.getAttribLocation("a_Position0");
		GLint colorHandleBBox = renderProgram_BBox.getAttribLocation("a_Color0");

		glBindBuffer(GL_ARRAY_BUFFER, vboBBox);

		glVertexAttribPointer(positionHandleBBox, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), NULL);
		glEnableVertexAttribArray(positionHandleBBox);

		glVertexAttribPointer(colorHandleBBox, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (char*)0 + 4 * sizeof(float));
		glEnableVertexAttribArray(colorHandleBBox);

		glDrawArrays(GL_LINES, 0, 3 * 8);

		glDisableVertexAttribArray(positionHandleBBox);
		glDisableVertexAttribArray(colorHandleBBox);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glUseProgram(0);
	}

#if !defined(HAVE_EGL)
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_PROGRAM_POINT_SIZE);
	//glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
#endif
}

void App::Terminate()
{
	DEALLOC(calculator);

	DEALLOC_ARR(initPos);
	DEALLOC_ARR(initCol);
	DEALLOC_ARR(initVel);
	
	renderProgram_BBox.clear();
	renderProgram_Particles.clear();

	CHECK_GL;
}

std::string App::GetName()
{
#ifdef COMPGL
	return std::string("comp_particles_gl");
#elif COMPCL
	return std::string("comp_particles_cl");
#elif COMPCU
	return std::string("comp_particles_cu");
#endif
}
