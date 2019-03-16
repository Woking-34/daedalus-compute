#include "glbase.h"

std::string openglGetErrorString(GLenum status)
{
	std::stringstream ss;

	switch (status)
	{
		case GL_INVALID_ENUM:
			ss << "GL_INVALID_ENUM";
			break;
		case GL_INVALID_VALUE:
			ss << "GL_INVALID_VALUE";
			break;
		case GL_INVALID_OPERATION:
			ss << "GL_INVALID_OPERATION";
			break;
		case GL_OUT_OF_MEMORY:
			ss << "GL_OUT_OF_MEMORY";
			break;
		default:
			ss << "GL_UNKNOWN_ERROR" << " - " << status;
			break;
	}

	return ss.str();
}