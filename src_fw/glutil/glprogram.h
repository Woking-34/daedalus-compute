#ifndef GLPROGRAM_H
#define GLPROGRAM_H

#include "glbase.h"
#include <string>

class GLProgram
{
public:
	GLProgram();
	~GLProgram();

	void setWGS(GLint x, GLint y, GLint z);

	void addHeader(const std::string& header);
	void addDefine(const std::string& define, GLenum shaderType);
	void addFile(const std::string& file, GLenum shaderType);
	void addString(const std::string& string, GLenum shaderType);

	void bindAttribLocation(GLuint location, const std::string& attribStr);
    void bindFragDataLocation(GLuint location, const std::string& uniformStr);
	
	void buildProgram();
	void useProgram();

	void clear();

	GLint* getAttribLocations()
	{
		return attribLocations;
	}

	GLuint getProgramObject()
	{
		return programObject;
	}

	GLint getAttribLocation(const std::string& attribStr) const;
	GLint getUniformLocation(const std::string& uniformStr) const;

	void setIntValue(int value, const char* parameter) const;
	void setFloatValue(float value, const char* parameter) const;

	void setFloatVector2(const float* vec2, const char* parameter) const;
	void setFloatVector4(const float* vec4, const char* parameter) const;

	void setFloatMatrix44(const float* mat44, const char* parameter) const;

	void printActiveUniforms();
    void printActiveAttribs();

protected:
	int getShaderType(int id)
	{
		switch(id)
		{
			case 0:
				return GL_VERTEX_SHADER;
				break;
			case 1:
				return GL_FRAGMENT_SHADER;
				break;
			case 2:
				return GL_GEOMETRY_SHADER;
				break;
			case 3:
				return GL_TESS_CONTROL_SHADER;
				break;
			case 4:
				return GL_TESS_EVALUATION_SHADER;
				break;
			case 5:
				return GL_COMPUTE_SHADER;
				break;
			default:
				// throw
				return -1;
		}
	}
	int getShaderID(GLenum shaderType)
	{
		switch(shaderType)
		{
			case GL_VERTEX_SHADER:
				return 0;
				break;
			case GL_FRAGMENT_SHADER:
				return 1;
				break;
			case GL_GEOMETRY_SHADER:
				return 2;
				break;
			case GL_TESS_CONTROL_SHADER:
				return 3;
				break;
			case GL_TESS_EVALUATION_SHADER:
				return 4;
				break;
			case GL_COMPUTE_SHADER:
				return 5;
				break;
			default:
				// throw
				return -1;
		}
	}
	int getShaderID(const std::string extStr)
	{
		if(extStr == "vert")
		{
			return 0;
		}
		else if(extStr == "frag")
		{
			return 1;
		}
		else if(extStr == "geom")
		{
			return 2;
		}
		else if(extStr == "tesc")
		{
			return 3;
		}
		else if(extStr == "tese")
		{
			return 4;
		}
		else if(extStr == "comp")
		{
			return 5;
		}
		else
		{
			// throw
			return -1;
		}
	}

	void checkShaderStatus(GLenum shaderType);
	void checkProgramStatus();
	
	static bool isGLES;

	std::string header;

	std::string fileNames[6];

	std::string extension[6];
	std::string defines[6];
	std::string sources[6];
	
	GLuint shaderObjects[6];
	GLuint programObject;

	// commonly used vertex attribs
	GLint attribLocations[8];
};

#endif