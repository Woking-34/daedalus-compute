#include <iostream>

#include "GLUTApplication.h"

int main(int argc, char** argv)
{
	GLUTApplication* m_pApp = GLUTApplication::GetApplication();

	if(m_pApp == NULL)
	{
		std::cout << "Failed to find Application instance!" << std::endl;
		return -1;
	}
	
	bool run = m_pApp->ParseCommandLineArgs(argc, argv);
	
	if(run)
	{
		m_pApp->InitGL(argc, argv);

		m_pApp->Initialize();

		m_pApp->RunGL();

		// called via glutCloseFunc
		//m_pApp->Terminate();
	}

	return 0;
}