#include "App_comp_cloth.h"

#ifdef COMPGL
	#include "COMPGL_CLOTH.h"
#elif COMPCL
	#include "COMPCL_CLOTH.h"
#elif COMPCU
	#include "COMPCU_CLOTH.h"
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

	vboID_InCurr = 0;
	vboID_InPrev = 0;
	vboID_OutCurr = 0;
	vboID_OutPrev = 0;
	vboID_Normals = 0;
	iboID = 0;

	{
		wgsX = 8;
		wgsY = 8;

		launchW = 256;
		launchH = 256;
	}

	{
		height = 4.25f;

		sizeX = 12.0f;
		sizeY = 12.0f;

		stepX = sizeX / (launchW - 1.0f);
		stepY = sizeY / (launchH - 1.0f);

		mass = 0.075f;
		damp = -0.0005f;
		dt = 1.0f / 60.0f;
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

	initPositions = new Vec4f[launchW*launchH];
	initNormals = new Vec4f[launchW*launchH];
	initIBO = new unsigned int[2 * 3 * (launchW - 1)*(launchH - 1)];

	initWeights = new float[launchW*launchH];

	stepX = sizeX / (launchW - 1.0f);
	stepY = sizeY / (launchH - 1.0f);

	// fill in positions
	for (int j = 0; j < launchH; ++j)
	{
		for (int i = 0; i < launchW; ++i)
		{
			initPositions[i + j*launchW] = Vec4f(-0.5f*sizeX + i*stepX, height, -0.5f*sizeY + j*stepY, 1.0f);
			initNormals[i + j*launchW] = Vec4f(0.0f, 1.0f, 0.0f, 0.0f);
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

	// fill in weights
	for (int j = 0; j < launchH; ++j)
	{
		for (int i = 0; i < launchW; ++i)
		{
			if (j == 0)
			{
				initWeights[i + j*launchW] = mass;
			}
			else
			{
				initWeights[i + j*launchW] = mass;
			}
		}
	}

	glGenBuffers(1, &vboID_InCurr);
	glBindBuffer(GL_ARRAY_BUFFER, vboID_InCurr);
	glBufferData(GL_ARRAY_BUFFER, launchW*launchH * 4 * sizeof(float), initPositions, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vboID_InPrev);
	glBindBuffer(GL_ARRAY_BUFFER, vboID_InPrev);
	glBufferData(GL_ARRAY_BUFFER, launchW*launchH * 4 * sizeof(float), initPositions, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vboID_OutCurr);
	glBindBuffer(GL_ARRAY_BUFFER, vboID_OutCurr);
	glBufferData(GL_ARRAY_BUFFER, launchW*launchH * 4 * sizeof(float), initPositions, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vboID_OutPrev);
	glBindBuffer(GL_ARRAY_BUFFER, vboID_OutPrev);
	glBufferData(GL_ARRAY_BUFFER, launchW*launchH * 4 * sizeof(float), initPositions, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vboID_Normals);
	glBindBuffer(GL_ARRAY_BUFFER, vboID_Normals);
	glBufferData(GL_ARRAY_BUFFER, launchW*launchH * 4 * sizeof(float), initNormals, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &iboID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * 3 * (launchW - 1)*(launchH - 1)*sizeof(unsigned int), initIBO, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	renderProgram_Velvet.addFile("velvet.vert", GL_VERTEX_SHADER);
	renderProgram_Velvet.addFile("velvet.frag", GL_FRAGMENT_SHADER);
	renderProgram_Velvet.buildProgram();

	const Vec4f eye(1.75f*5, 1.75f*5, 1.75f*5, 1.0f);
	const Vec4f center(0.0f, 0.0f, 0.0f, 1.0f);
	const Vec4f up(0.0f, 1.0f, 0.0f, 0.0f);

	mainCamera.setViewParams(eye, center, up);
	mainCamera.setProjParams(60.0f, (float)(currWidth) / (float)(currHeight), 0.25f, 100.0f);
	mainCamera.updateRays();

	// init gl state
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	CHECK_GL;

#ifdef COMPGL
	api = "GL";
	calculator = new COMPGL_CLOTH;
#elif COMPCL
	api = "CL";
	calculator = new COMPCL_CLOTH;
#elif COMPCU
	api = "CU";
	calculator = new COMPCU_CLOTH;
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
	calculator->mass = mass;
	calculator->damp = damp;
	calculator->dt = dt;

	calculator->useInterop = useInterop;
	calculator->bufferPositions = (float*)initPositions;
	calculator->bufferNormals = (float*)initNormals;
	calculator->bufferWeights = initWeights;
	calculator->vboInCurr = vboID_InCurr;
	calculator->vboInPrev = vboID_InPrev;
	calculator->vboOutCurr = vboID_OutCurr;
	calculator->vboOutPrev = vboID_OutPrev;
	calculator->vboNormals = vboID_Normals;

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
	float translationScale = 0.0015f * delta_time;
	float rotationScale = 0.0015f;

	if (isSHIFTPressed)
	{
		translationScale *= 10.0f;
	}

	if (normalKeyState['w'] || normalKeyState['W'])
	{
		mainCamera.updateTranslationForwardBackward(-translationScale);
	}

	if (normalKeyState['s'] || normalKeyState['S'])
	{
		mainCamera.updateTranslationForwardBackward(translationScale);
	}

	if (normalKeyState['a'] || normalKeyState['A'])
	{
		mainCamera.updateTranslationLeftRight(-translationScale);
	}

	if (normalKeyState['d'] || normalKeyState['D'])
	{
		mainCamera.updateTranslationLeftRight(translationScale);
	}

	if (normalKeyState['q'] || normalKeyState['Q'])
	{
		mainCamera.updateTranslationUpDown(-translationScale);
	}

	if (normalKeyState['e'] || normalKeyState['E'])
	{
		mainCamera.updateTranslationUpDown(translationScale);
	}

	if (mouseState[GLUT_LEFT_BUTTON])
	{
		if (mouseX_old != -1 && mouseY_old != -1)
		{
			mainCamera.updateRotation(rotationScale * (mouseX - mouseX_old), rotationScale * (mouseY - mouseY_old), 0.0f);
		}

		mouseX_old = mouseX;
		mouseY_old = mouseY;
	}
	else
	{
		mouseX_old = -1;
		mouseY_old = -1;
	}
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
				glBindBuffer(GL_ARRAY_BUFFER, calculator->vboInCurr);
				void* hostPtr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
				memcpy(hostPtr, calculator->bufferPositions, launchW*launchH * 4 * sizeof(float));
				glUnmapBuffer(GL_ARRAY_BUFFER);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}
			{
				glBindBuffer(GL_ARRAY_BUFFER, calculator->vboNormals);
				void* hostPtr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
				memcpy(hostPtr, calculator->bufferNormals, launchW*launchH * 4 * sizeof(float));
				glUnmapBuffer(GL_ARRAY_BUFFER);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}
		}
	}

	mainCamera.setProjParams(60.0f, (float)(currWidth) / (float)(currHeight), 0.15f, 100.0f);
	mainCamera.updateRays();

	Mat44f matModel = createTranslationMatrix(0.0f, 0.0f, 0.0f);
	Mat44f matMV = matModel * mainCamera.viewMat;
	Mat44f matMVP = matModel * mainCamera.viewMat * mainCamera.projMat;

	glViewport(0, 0, currWidth, currHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	CHECK_GL;

	if(1)
	{
		renderProgram_Velvet.useProgram();

		renderProgram_Velvet.setFloatMatrix44(matMV, "u_MVMat");
		renderProgram_Velvet.setFloatMatrix44(matMVP, "u_MVPMat");

		GLint positionHandleVelvet = renderProgram_Velvet.getAttribLocation("a_Position0");
		GLint normalHandleVelvet = renderProgram_Velvet.getAttribLocation("a_Normal0");

		renderProgram_Velvet.setFloatVector4(Vec4f(1.0f, 0.0225f, 0.0f, 1.0f), "base_color");
		renderProgram_Velvet.setFloatVector4(Vec4f(0.8118f, 0.3940f, 0.3361f, 1.0f), "sheen");
		renderProgram_Velvet.setFloatVector4(Vec4f(0.1255f, 0.1267f, 0.1267f, 1.0f), "shiny");

		renderProgram_Velvet.setFloatValue(0.1f, "roughness");
		renderProgram_Velvet.setFloatValue(20.5f, "edginess");
		renderProgram_Velvet.setFloatValue(0.1f, "backscatter");

		renderProgram_Velvet.setFloatVector4(Vec4f(0.5f, 0.5f, 0.5f, 1.0f), "global_ambient");
		renderProgram_Velvet.setFloatVector4(Vec4f(0.2333f, 0.2333f, 0.2333f, 1.0f), "Ka");
		renderProgram_Velvet.setFloatVector4(Vec4f(0.4314f, 0.4105f, 0.4052f, 1.0f), "Kd");

		renderProgram_Velvet.setFloatVector4(Vec4f(10.0f, 10.0f, 10.0f, 1.0f), "light_pos");
		renderProgram_Velvet.setFloatVector4(Vec4f(1.0f, 1.0f, 1.0f, 1.0f), "light_color");

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboID);

		glBindBuffer(GL_ARRAY_BUFFER, calculator->vboInCurr);
		glVertexAttribPointer(positionHandleVelvet, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glBindBuffer(GL_ARRAY_BUFFER, calculator->vboNormals);
		glVertexAttribPointer(normalHandleVelvet, 4, GL_FLOAT, GL_FALSE, 0, NULL);

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
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(mainCamera.projMat);

		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(mainCamera.viewMat);

		// DrawCloth()
		{
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboID);

			glBindBuffer(GL_ARRAY_BUFFER, calculator->vboInCurr);
			glVertexPointer(4, GL_FLOAT, 0, 0);

			glBindBuffer(GL_ARRAY_BUFFER, calculator->vboNormals);
			glColorPointer(4, GL_FLOAT, 0, 0);

			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_COLOR_ARRAY);

			glDrawElements(GL_TRIANGLES, 2 * 3 * (launchW - 1)*(launchH - 1), GL_UNSIGNED_INT, 0);

			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_COLOR_ARRAY);
		}

		// DrawGrid()
		{
			const int GRID_SIZE = 6;

			glBegin(GL_LINES);
			glColor3f(1.0f, 1.0f, 1.0f);
			for (int i = -GRID_SIZE; i <= GRID_SIZE; i++)
			{
				glVertex3f((float)i, 0, (float)-GRID_SIZE);
				glVertex3f((float)i, 0, (float)GRID_SIZE);

				glVertex3f((float)-GRID_SIZE, 0, (float)i);
				glVertex3f((float)GRID_SIZE, 0, (float)i);
			}
			glEnd();
		}

		{
			//glTranslatef(0.0f, 2.0f, 0.0f);
			//drawSphere(1.75f, 20);
		}

		CHECK_GL;
	}
}

void App::Terminate()
{
	calculator->terminate();
	DEALLOC(calculator);

	DEALLOC_ARR(initPositions);
	DEALLOC_ARR(initNormals);
	DEALLOC_ARR(initIBO);
	
	DEALLOC_ARR(initWeights);

	renderProgram_Velvet.clear();

	CHECK_GL;
}

std::string App::GetName()
{
#ifdef COMPGL
	return std::string("comp_cloth_gl");
#elif COMPCL
	return std::string("comp_cloth_cl");
#elif COMPCU
	return std::string("comp_cloth_cu");
#endif
}
