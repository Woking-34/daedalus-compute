#include "GLUTApplication.h"

#include "system/log.h"
#include "system/filesystem.h"

#include "math/util/camera.h"

GLUTApplication* GLUTApplication::sApplication = NULL;

GLUTApplication::GLUTApplication()
{
	sApplication = this;

	glutWindowHandle = 0;
	frameCounter = 0;
	isFullscreen = false;

	memset( normalKeyState, 0, sizeof( normalKeyState ) );
	memset( specialKeyState, 0, sizeof( specialKeyState ) );
	memset( mouseState, 0, sizeof( mouseState ) );

	isENTERPressed = false;
	isSHIFTPressed = false;
	isALTPressed = false;

	prev_time = curr_time = delta_time = 0;
	mouseX = mouseY = 0;
	mouseX_old = mouseY_old = 0;

	currWidth = 800;
	currHeight = 800;

	isWire = false;

	appCamera = nullptr;
	translationScale = 0.0015f;
	rotationScale = 0.0015f;
}

GLUTApplication::~GLUTApplication()
{

}

GLUTApplication* GLUTApplication::GetApplication()
{
	return sApplication;
}

bool GLUTApplication::ParseCommandLineArgs(int argc, char** argv)
{
	if(argc == 2)
	{
		std::string nextarg(argv[1]);

		if(nextarg == "--help")
		{
			PrintCommandLineHelp();

			return false;
		}
	}

	for(int i = 1; i < argc; ++i)
	{
		std::string nextarg(argv[i]);

		if(nextarg == "--ei" && i+2 < argc)
		{
			rawi[argv[i+1]] = (int)atol(argv[i+2]);
			i+=2;
		}
		else if(nextarg == "--ef" && i+2 < argc)
		{
			rawf[argv[i+1]] = (float)atof(argv[i+2]);
			i+=2;
		}
		else if (nextarg == "--ez" && i+2 < argc)
		{
			std::string sval = argv[i+2];

			if (sval == "true" || sval == "1") {
				rawb[argv[i+1]] = true;
			} else if (sval == "false" || sval == "0") {
				rawb[argv[i+1]] = false;
			} else {
				LOG_WARN( LogLine() << "Failed to parse " << sval << "\n" );
			}

			i+=2;
		}
		else if(nextarg == "--es" && i+2 < argc)
		{
			raws[argv[i+1]] = argv[i+2];
			i+=2;
		}
	}

	return true;
}

void GLUTApplication::PrintCommandLineHelp()
{
	LOG( "Usage" );
	LOG( "\t[--ei|--ef|--es|--ez <EXTRA_KEY> <EXTRA_VALUE>]" );
	LOG( "" );
	LOG( "\t--ei key int/uint" );
	LOG( "\t--ef key float" );
	LOG( "\t--es key string" );
	LOG( "\t--ez key bool" );
	LOG( "" );
}

void GLUTApplication::InitGlut(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - currWidth/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - currHeight/2);
	glutInitWindowSize(currWidth, currHeight);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_STENCIL);

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	
	glutWindowHandle = glutCreateWindow( GetName().c_str() );
	glutIgnoreKeyRepeat(1);

	LOG_BOOL(glutWindowHandle != 0, "InitGlut()");
}

void GLUTApplication::InitGlew()
{
	GLenum glewInitResult = glewInit();

	while(glGetError() != GL_NO_ERROR) {}

	LOG_BOOL(glewInitResult == GLEW_OK, "InitGlew()");
}

void GLUTApplication::PrintGlinfo()
{
	glVendor = std::string( (const char*)glGetString(GL_VENDOR) );
	glRenderer = std::string( (const char*)glGetString(GL_RENDERER) );
	glVersion = std::string( (const char*)glGetString(GL_VERSION) );
	glslVersion = std::string( (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION) );

	glewVersion = std::string( (const char*)glewGetString(GLEW_VERSION) );

	glGetIntegerv(GL_MAJOR_VERSION, &glMajor);
	glGetIntegerv(GL_MINOR_VERSION, &glMinor);

	GLint n;
	glGetIntegerv(GL_NUM_EXTENSIONS, &n);
	for (int i = 0; i < n; i++)
	{
		glExtensionVec.push_back(std::string((const char*)glGetStringi(GL_EXTENSIONS, i)));
	}
	std::sort(glExtensionVec.begin(), glExtensionVec.end());   
	
	LOG( tabulateStrings("GL_VENDOR:", glVendor) );
	LOG( tabulateStrings("GL_RENDERER:", glRenderer) );
	LOG( tabulateStrings("GL_VERSION:", glVersion) );
	LOG( tabulateStrings("GLSL_VERSION:", glslVersion) );

	LOG( tabulateStrings("GLEW_VERSION:", glewVersion) );
}

bool GLUTApplication::SetSwapInterval(int interval)
{
#ifdef BUILD_WINDOWS
	// https://stackoverflow.com/questions/589064/how-to-enable-vertical-sync-in-opengl/589232
	// This is pointer to function which returns pointer to string with list of all wgl extensions
	PFNWGLGETEXTENSIONSSTRINGEXTPROC _wglGetExtensionsStringEXT = NULL;

	// Determine pointer to wglGetExtensionsStringEXT function
	_wglGetExtensionsStringEXT = (PFNWGLGETEXTENSIONSSTRINGEXTPROC)wglGetProcAddress("wglGetExtensionsStringEXT");

	if (strstr(_wglGetExtensionsStringEXT(), "WGL_EXT_swap_control") != NULL)
	{
		PFNWGLSWAPINTERVALEXTPROC       wglSwapIntervalEXT;
		PFNWGLGETSWAPINTERVALEXTPROC    wglGetSwapIntervalEXT;

		// Extension is supported, init pointers.
		wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
		wglGetSwapIntervalEXT = (PFNWGLGETSWAPINTERVALEXTPROC)wglGetProcAddress("wglGetSwapIntervalEXT");

		wglSwapIntervalEXT(interval);

		return true;
	}
#endif

//#ifdef BUILD_UNIX
//	// http://ludobloom.com/svn/StemLibProjects/glutshell/trunk/source/glutshell/GLUTShell.c
//	const char * extensions = glXQueryExtensionsString(glXGetCurrentDisplay(), 0);
//	if (strstr(extensions, "GLX_EXT_swap_control")) {
//		PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC)glXGetProcAddress((const GLubyte *) "glXSwapIntervalEXT");
//		glXSwapIntervalEXT(glXGetCurrentDisplay(), glXGetCurrentDrawable(), sync);
//
//		return true;
//	}
//	else if (strstr(extensions, "GLX_SGI_swap_control")) {
//		PFNGLXSWAPINTERVALSGIPROC glxSwapIntervalSGI = (PFNGLXSWAPINTERVALSGIPROC)glXGetProcAddress((const GLubyte *) "glXSwapIntervalSGI");
//		glxSwapIntervalSGI(interval);
//
//		return true;
//	}
//#endif

	return false;
}

void GLUTApplication::InitGL(int argc, char** argv)
{
	FileSystem::Init();

	InitGlut(argc, argv);
	InitGlew();

	LOG("");
	PrintGlinfo();
	LOG("");

	SetSwapInterval(0);

	return;

	int res = 0;

	glGetIntegerv(GL_MAJOR_VERSION, &res);
	glGetIntegerv(GL_MINOR_VERSION, &res);

	glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &res);
	printf("Max Color Attachments: %d\n", res);

	glGetIntegerv(GL_MAX_FRAMEBUFFER_WIDTH, &res);
	printf("Max Framebuffer Width: %d\n", res);

	glGetIntegerv(GL_MAX_FRAMEBUFFER_HEIGHT, &res);
	printf("Max Framebuffer Height: %d\n", res);

	glGetIntegerv(GL_MAX_FRAMEBUFFER_SAMPLES, &res);
	printf("Max Framebuffer Samples: %d\n", res);
}

void GLUTApplication::RunGL()
{
	::glutIdleFunc(&GLUTApplication::glutIdleFunc);
	::glutDisplayFunc(&GLUTApplication::glutDisplayFunc);
	::glutReshapeFunc(&GLUTApplication::glutReshapeFunc);

	::glutKeyboardFunc(&GLUTApplication::glutKeyboardFunc);
	::glutKeyboardUpFunc(&GLUTApplication::glutKeyboardUpFunc);

	::glutSpecialFunc(&GLUTApplication::glutKeyboardSpecialFunc);
	::glutSpecialUpFunc(&GLUTApplication::glutKeyboardSpecialUpFunc);

	::glutMouseFunc(&GLUTApplication::glutMouseFunc);
	::glutMotionFunc(&GLUTApplication::glutMotionFunc);

	::glutCloseFunc(&GLUTApplication::glutCloseFunction);
	::glutTimerFunc(0, &GLUTApplication::glutFPSFunc, 0);

	glutMainLoop();
}

void GLUTApplication::OnIdle()
{
	prev_time = curr_time;
	curr_time = glutGet(GLUT_ELAPSED_TIME);
	delta_time = curr_time - prev_time;

	if (isALTPressed && isENTERPressed)
	{
		glutFullScreenToggle();
		isFullscreen = !isFullscreen;

		isENTERPressed = false;

		LOG( LogLine() << "Window resolution: " << glutGet(GLUT_WINDOW_WIDTH) << " x " << glutGet(GLUT_WINDOW_HEIGHT) )
	}

	if(specialKeyState[GLUT_KEY_INSERT])
	{
		specialKeyState[GLUT_KEY_INSERT] = false;
		
		isWire = !isWire;

		if (isWire)
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		}
		else
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}
	}

	float _translationScale = translationScale * delta_time;

	if(isSHIFTPressed)
	{
		_translationScale *= 5.0f;
	}

	if(appCamera)
	{
		if (normalKeyState['w'] || normalKeyState['W'])
		{
			appCamera->updateTranslationForwardBackward(-_translationScale);
		}

		if (normalKeyState['s'] || normalKeyState['S'])
		{
			appCamera->updateTranslationForwardBackward(_translationScale);
		}

		if (normalKeyState['a'] || normalKeyState['A'])
		{
			appCamera->updateTranslationLeftRight(-_translationScale);
		}

		if (normalKeyState['d'] || normalKeyState['D'])
		{
			appCamera->updateTranslationLeftRight(_translationScale);
		}

		if (normalKeyState['q'] || normalKeyState['Q'])
		{
			appCamera->updateTranslationUpDown(-_translationScale);
		}

		if (normalKeyState['e'] || normalKeyState['E'])
		{
			appCamera->updateTranslationUpDown(_translationScale);
		}

		if (mouseState[GLUT_LEFT_BUTTON])
		{
			if (mouseX_old != -1 && mouseY_old != -1)
			{
				appCamera->updateRotation(rotationScale * (mouseX - mouseX_old), rotationScale * (mouseY - mouseY_old), 0.0f);
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

	Update();

	glutPostRedisplay();
}

void GLUTApplication::OnDisplay()
{
	++frameCounter;

	Render();
	
	glFlush();
	glFinish();
	glutSwapBuffers();

	CHECK_GL;

	if(normalKeyState['*'])
	{
		normalKeyState['*'] = false;

		uchar* buffer = new uchar[currWidth * currHeight * 4];
		
		glReadBuffer( GL_BACK );

		glReadPixels( 0, 0, currWidth, currHeight, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)buffer);

		std::fstream fh( (FileSystem::GetRootFolder() + GetName() +  ".ppm").c_str(), std::fstream::out | std::fstream::binary );
	
		fh << "P6\n";
		fh << currWidth << "\n" << currHeight << "\n" << 0xff << std::endl;

		for(int j = 0; j < currHeight; ++j) 
		{
			for(int i = 0; i < currWidth; ++i) 
			{
				// flip ogl backbuffer vertically
				fh << buffer[4*((currHeight-1-j)*currWidth+i)+0];
				fh << buffer[4*((currHeight-1-j)*currWidth+i)+1];
				fh << buffer[4*((currHeight-1-j)*currWidth+i)+2];
			}
		}
		fh.flush();
		fh.close();

		delete[] buffer;
	}
}

void GLUTApplication::OnReshape(int w, int h)
{
	currWidth = w;
	currHeight = h;
}

void GLUTApplication::ProcessNormalKeysPress(unsigned char key, int x, int y)
{
	if( key >= 'A' && key <= 'Z')
	{
		key += 32;
	}
	if (key == 13)
	{
		isENTERPressed = true;
	}

	normalKeyState[key] = true;

	// esc is pressed
	if(normalKeyState[27])
	{
		glutLeaveMainLoop();
	}
}

void GLUTApplication::ProcessNormalKeysRelease(unsigned char key, int x, int y)
{
	if (key >= 'A' && key <= 'Z')
	{
		key += 32;
	}
	if (key == 13)
	{
		isENTERPressed = false;
	}

	normalKeyState[key] = false;
}

void GLUTApplication::ProcessSpecialKeysPress(int key, int x, int y)
{
	//GLUT_ACTIVE_SHIFT
	//GLUT_ACTIVE_CTRL
	//GLUT_ACTIVE_ALT
	//int mod = glutGetModifiers();

	if( key == 112 )
	{
		isSHIFTPressed = true;
	}
	if (key == 116)
	{
		isALTPressed = true;
	}

	specialKeyState[key] = true;
}

void GLUTApplication::ProcessSpecialKeysRelease(int key, int x, int y)
{
	//GLUT_ACTIVE_SHIFT
	//GLUT_ACTIVE_CTRL
	//GLUT_ACTIVE_ALT
	//int mod = glutGetModifiers();

	if( key == 112 )
	{
		isSHIFTPressed = false;
	}
	if (key == 116)
	{
		isALTPressed = false;
	}

	specialKeyState[key] = false;
}

void GLUTApplication::ProcessMouseClicks(int button, int state, int x, int y)
{
	mouseX = x;
	mouseY = y;

	mouseState[button] = state == GLUT_DOWN;
}

void GLUTApplication::ProcessActiveMouseMove(int x, int y)
{
	mouseX_old = mouseX;
	mouseY_old = mouseY;

	mouseX = x;
	mouseY = y;
}

void GLUTApplication::OnClose(void)
{
	sApplication->Terminate();
}

void GLUTApplication::OnFPS(int value)
{
	std::stringstream titleSS;
	titleSS << GetName() << " - " << delta_time << " ms / " << 4*frameCounter << " FPS @ " << currWidth << " x " << currHeight;

	frameCounter = 0;

	glutSetWindowTitle(titleSS.str().c_str());
	glutTimerFunc(250, &GLUTApplication::glutFPSFunc, 1);
}

void GLUTApplication::glutIdleFunc()
{
	sApplication->OnIdle();
}

void GLUTApplication::glutDisplayFunc()
{
	sApplication->OnDisplay();
}

void GLUTApplication::glutReshapeFunc(int w, int h)
{
	sApplication->OnReshape(w, h);
}

void GLUTApplication::glutKeyboardFunc(unsigned char key, int x, int y)
{
	sApplication->ProcessNormalKeysPress(key,x,y);
}

void GLUTApplication::glutKeyboardUpFunc(unsigned char key, int x, int y)
{
	sApplication->ProcessNormalKeysRelease(key,x,y);
}

void GLUTApplication::glutKeyboardSpecialFunc(int key, int x, int y)
{
	sApplication->ProcessSpecialKeysPress(key,x,y);
}

void GLUTApplication::glutKeyboardSpecialUpFunc(int key, int x, int y)
{
	sApplication->ProcessSpecialKeysRelease(key,x,y);
}

void GLUTApplication::glutMouseFunc(int button, int state, int x, int y)
{
	sApplication->ProcessMouseClicks(button, state, x, y);
}

void GLUTApplication::glutMotionFunc(int x, int y)
{
	sApplication->ProcessActiveMouseMove(x, y);
}

void GLUTApplication::glutCloseFunction(void)
{
	sApplication->OnClose();
}

void GLUTApplication::glutFPSFunc(int value)
{
	sApplication->OnFPS(value);
}