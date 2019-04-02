#include "App_comp_deform.h"

#ifdef COMPGL
	#include "COMPGL_DEFORM.h"
#elif COMPCL
	#include "COMPCL_DEFORM.h"
#elif COMPCU
	#include "COMPCU_DEFORM.h"
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

	vboID = 0;
	iboID = 0;

	initVBO = NULL;
	initIBO = NULL;

	{
		wgsX = 8;
		wgsY = 8;

		launchW = 128;
		launchH = 128;

		launchW = 256;
		launchH = 256;
	}

	{
		deformMode = 0;

		sizeX = 12.0f;
		sizeY = 12.0f;

		stepX = sizeX / (launchW - 1.0f);
		stepY = sizeY / (launchH - 1.0f);
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

	ReadCmdArg(STRING(launchW), launchW);
	ReadCmdArg(STRING(launchH), launchH);
	ReadCmdArg(STRING(wgsX), wgsX);
	ReadCmdArg(STRING(wgsY), wgsY);

	initVBO = new Vec4f[2*launchW*launchH];
	initIBO = new unsigned int[2 * 3 * (launchW - 1)*(launchH - 1)];

	stepX = sizeX / (launchW - 1.0f);
	stepY = sizeY / (launchH - 1.0f);

	// fill in vertices
	for (int j = 0; j < launchH; ++j)
	{
		for (int i = 0; i < launchW; ++i)
		{
			initVBO[2 * (i + j*launchW) + 0] = Vec4f(-0.5f*sizeX + i*stepX, 0.0f, -0.5f*sizeY + j*stepY, 1.0f);
			initVBO[2 * (i + j*launchW) + 1] = Vec4f(0.0f, 1.0f, 0.0f, 0.0f);
		}
	}

	// fill in indices
	unsigned int* id = &initIBO[0];
	for (int j = 0; j < launchH-1; ++j)
	{
		for (int i = 0; i < launchW-1; ++i)
		{
			int i0 = j * (launchW)+i;
			int i1 = i0 + 1;
			int i2 = i0 + (launchW);
			int i3 = i2 + 1;
			if ((i + j) % 2) {
				*id++ = i0; *id++ = i2; *id++ = i1;
				*id++ = i1; *id++ = i2; *id++ = i3;
			}
			else {
				*id++ = i0; *id++ = i2; *id++ = i3;
				*id++ = i0; *id++ = i3; *id++ = i1;
			}
		}
	}

	glGenBuffers(1, &vboID);
	glBindBuffer(GL_ARRAY_BUFFER, vboID);
	glBufferData(GL_ARRAY_BUFFER, 2 * launchW*launchH * 4 * sizeof(float), initVBO, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &iboID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * 3 * (launchW - 1)*(launchH - 1)*sizeof(unsigned int), initIBO, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	renderProgram.addFile("lightpoint_meshcolor.vert", GL_VERTEX_SHADER);
	renderProgram.addFile("lightpoint_meshcolor.frag", GL_FRAGMENT_SHADER);
	renderProgram.buildProgram();

	const Vec4f eye(1.75f*5, 1.75f*5, 1.75f*5, 1.0f);
	const Vec4f center(0.0f, 0.0f, 0.0f, 1.0f);
	const Vec4f up(0.0f, 1.0f, 0.0f, 0.0f);

	mainCamera.setViewParams(eye, center, up);
	mainCamera.setProjParams(60.0f, (float)(currWidth) / (float)(currHeight), 0.25f, 100.0f);
	mainCamera.updateRays();

	// init gl state
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	appCamera = &mainCamera;

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	LOG(LogLine() << "Press F1-F3 to switch between different scenes!\n");

	CHECK_GL;

#ifdef COMPGL
	api = "GL";
	calculator = new COMPGL_DEFORM;
#elif COMPCL
	api = "CL";
	calculator = new COMPCL_DEFORM;
#elif COMPCU
	api = "CU";
	calculator = new COMPCU_DEFORM;
#endif

	calculator->sizeX = sizeX;
	calculator->sizeY = sizeY;

	calculator->launchW = launchW;
	calculator->launchH = launchH;

	calculator->wgsX = wgsX;
	calculator->wgsY = wgsY;

	calculator->sizeX = sizeX;
	calculator->sizeY = sizeY;
	calculator->stepX = stepX;
	calculator->stepY = stepY;

	calculator->useInterop = useInterop;
	calculator->bufferVertices = (float*)initVBO;
	calculator->vbo = vboID;

	calculator->useCLGDevice = useCLGDevice;
	calculator->useCLPId = useCLPId;
	calculator->useCLDId = useCLDId;

	calculator->init();

	// refresh interop status based on device override
	useInterop = calculator->useInterop;

	LOG( LogLine() << "GLOBAL_WORK_SIZE: " << calculator->launchW << " x " << calculator->launchH << "\n" );
	LOG( LogLine() << "LOCAL_WORK_SIZE: " << calculator->wgsX << " x " << calculator->wgsY << "\n" );
	LOG( LogLine() << "INTEROP: " << (useInterop ? "ENABLED" : "DISABLED") << "\n" );
}

void App::Update()
{
	if(specialKeyState[GLUT_KEY_F1])
	{
		specialKeyState[GLUT_KEY_F1] = false;
		deformMode = 0;
	}

	if(specialKeyState[GLUT_KEY_F2])
	{
		specialKeyState[GLUT_KEY_F2] = false;
		deformMode = 1;
	}

	if(specialKeyState[GLUT_KEY_F3])
	{
		specialKeyState[GLUT_KEY_F3] = false;
		deformMode = 2;
	}
}

void App::Render()
{
	CHECK_GL;

	{
		// compute
		{
			calculator->compute(deformMode, curr_time * 1.0f);
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
				glBindBuffer(GL_ARRAY_BUFFER, calculator->vbo);
				void* hostPtr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
				memcpy(hostPtr, calculator->bufferVertices, 2 * launchW*launchH * 4 * sizeof(float));
				glUnmapBuffer(GL_ARRAY_BUFFER);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}
		}
	}
	
	mainCamera.setProjParams(60.0f, (float)(currWidth) / (float)(currHeight), 0.15f, 100.0f);
	mainCamera.updateRays();

	Mat44f matM = createTranslationMatrix(0.0f, 0.0f, 0.0f);
	Mat44f matMV = matM * mainCamera.viewMat;
	Mat44f matMVP = matM * mainCamera.viewMat * mainCamera.projMat;

	glViewport(0, 0, currWidth, currHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	CHECK_GL;

	if(1)
	{
		renderProgram.useProgram();

		renderProgram.setFloatMatrix44(matM, "u_MMat");
		renderProgram.setFloatMatrix44(matMVP, "u_MVPMat");

		GLint positionHandleVelvet = renderProgram.getAttribLocation("a_Position0");
		GLint normalHandleVelvet = renderProgram.getAttribLocation("a_Normal0");

		renderProgram.setFloatVector4(mainCamera.getPosition(), "eyePosW");
		renderProgram.setFloatVector4(Vec4f(5.0f, 5.0f, 5.0f, 1.0f), "lightPosW");

		if(deformMode == 0)
		{
			renderProgram.setFloatVector4(Vec4f(1.0f, 0.0f, 0.0f, 1.0f), "diffMaterial");
		}

		if(deformMode == 1)
		{
			renderProgram.setFloatVector4(Vec4f(0.0f, 1.0f, 0.0f, 1.0f), "diffMaterial");
		}

		if(deformMode == 2)
		{
			renderProgram.setFloatVector4(Vec4f(0.0f, 0.0f, 1.0f, 1.0f), "diffMaterial");
		}

		renderProgram.setFloatVector4(Vec4f(1.0f, 1.0f, 1.0f, 10.0f), "specMaterial");
		renderProgram.setFloatVector4(Vec4f(0.25f, 0.25f, 0.25f, 1.0f), "ambMaterial");
		renderProgram.setFloatVector4(Vec4f(0.0f, 64.0f, 0.0f, 0.0f), "attenLight");

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboID);

		glBindBuffer(GL_ARRAY_BUFFER, calculator->vbo);
		glVertexAttribPointer(positionHandleVelvet, 4, GL_FLOAT, GL_FALSE, 8*sizeof(float), (char*)0 + (0*sizeof(float)));
		glVertexAttribPointer(normalHandleVelvet, 4, GL_FLOAT, GL_FALSE, 8*sizeof(float), (char*)0 + (4*sizeof(float)));

		glEnableVertexAttribArray(positionHandleVelvet);
		glEnableVertexAttribArray(normalHandleVelvet);

		glDrawElements(GL_TRIANGLES, 2 * 3 * (launchW - 1)*(launchH - 1), GL_UNSIGNED_INT, 0);

		glDisableVertexAttribArray(positionHandleVelvet);
		glDisableVertexAttribArray(normalHandleVelvet);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glUseProgram(0);

		CHECK_GL;
	}
	else
	{
		glUseProgram(0);

		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(mainCamera.projMat);

		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(mainCamera.viewMat);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboID);

		glBindBuffer(GL_ARRAY_BUFFER, calculator->vbo);

		glVertexPointer(4, GL_FLOAT, 8*sizeof(float), ((char *)0 + 0*sizeof(float)));
		glColorPointer(4, GL_FLOAT, 8*sizeof(float), ((char *)0 + 4*sizeof(float)));

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		glDrawElements(GL_TRIANGLES, 2 * 3 * (launchW - 1)*(launchH - 1), GL_UNSIGNED_INT, 0);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);

		CHECK_GL;
	}
}

void App::Terminate()
{
	DEALLOC(calculator);

	DEALLOC_ARR(initIBO);
	
	renderProgram.clear();

	CHECK_GL;
}

std::string App::GetName()
{
#ifdef COMPGL
	return std::string("comp_deform_gl");
#elif COMPCL
	return std::string("comp_deform_cl");
#elif COMPCU
	return std::string("comp_deform_cu");
#endif
}
