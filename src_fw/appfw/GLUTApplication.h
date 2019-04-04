#ifndef GLUTAPPLICATION_H
#define GLUTAPPLICATION_H

#include "system/platform.h"
#include "glutil/glbase.h"

class Camera;

class GLUTApplication
{
public:
	GLUTApplication();
	virtual ~GLUTApplication();

	static GLUTApplication* GetApplication( );

	virtual bool ParseCommandLineArgs(int argc, char** argv);
	virtual void PrintCommandLineHelp();

	virtual void Initialize() = 0;
	virtual void Update() = 0;
	virtual void Render() = 0;
	virtual void Terminate() = 0;

	void InitGL(int argc, char** argv);
	void RunGL();

	virtual std::string GetName() = 0;

protected:
	int glutWindowHandle;
	int frameCounter;
	bool isFullscreen;

	std::map<std::string, int> rawi;
	std::map<std::string, uint> rawui;
	std::map<std::string, float> rawf;
	std::map<std::string, double> rawd;
	std::map<std::string, bool> rawb;
	std::map<std::string, std::string> raws;

	void ReadCmdArg(const std::string& key, bool& value)
	{
		if(rawb.count(key) == 1)
		{
			value = rawb[key];
		}
	}

	void ReadCmdArg(const std::string& key, int& value)
	{
		if(rawi.count(key) == 1)
		{
			value = rawi[key];
		}
	}

	int currWidth, currHeight;
	bool isWire;

	Camera* appCamera;
	float translationScale, rotationScale;
	
	bool normalKeyState[256], specialKeyState[256], mouseState[3];
	bool isENTERPressed, isSHIFTPressed, isALTPressed;
	int mouseX, mouseY, mouseX_old, mouseY_old;

	int prev_time, curr_time, delta_time;

	std::string glVendor;
	std::string glRenderer;
	std::string glVersion;
	std::string glslVersion;
	std::string glewVersion;

	GLint glMajor, glMinor;
	std::vector<std::string> glExtensionVec;

	bool hasGLExtension(const std::string glextToken)
	{
		for(unsigned int i = 0; i < glExtensionVec.size(); ++i)
		{
			if( glextToken == glExtensionVec[i] )
			{
				return true;
			}
		}

		return false;
	}

	void InitGlut(int argc, char** argv);
	void InitGlew();
	void PrintGlinfo();

	bool SetSwapInterval(int interval);

	virtual void OnIdle();
	virtual void OnDisplay();
	virtual void OnReshape(int w, int h);

	virtual void ProcessNormalKeysPress(unsigned char key, int x, int y);
	virtual void ProcessNormalKeysRelease(unsigned char key, int x, int y);
	virtual void ProcessSpecialKeysPress(int key, int x, int y);
	virtual void ProcessSpecialKeysRelease(int key, int x, int y);
	virtual void ProcessMouseClicks(int button, int state, int x, int y);
	virtual void ProcessActiveMouseMove(int x, int y);

	void OnClose(void);
	void OnFPS(int);

	static void glutIdleFunc();
	static void glutDisplayFunc();
	static void glutReshapeFunc(int w, int h);

	static void glutKeyboardFunc(unsigned char key, int x, int y);
	static void glutKeyboardUpFunc(unsigned char key, int x, int y);
	static void glutKeyboardSpecialFunc(int key, int x, int y);
	static void glutKeyboardSpecialUpFunc(int key, int x, int y);
	static void glutMouseFunc(int button, int state, int x, int y);
	static void glutMotionFunc(int x, int y);

	static void glutCloseFunction(void);
	static void glutFPSFunc(int);

	static GLUTApplication* sApplication;
};


#endif // GLUTAPPLICATION_H