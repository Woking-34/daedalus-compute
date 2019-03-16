#include "glprogram.h"

#include "system/platform.h"
#include "system/filesystem.h"
#include "system/log.h"

bool GLProgram::isGLES = false;

GLProgram::GLProgram()
{
	shaderObjects[0] = 0;
	shaderObjects[1] = 0;
	shaderObjects[2] = 0;
	shaderObjects[3] = 0;
	shaderObjects[4] = 0;
	shaderObjects[5] = 0;

	programObject = 0;

	attribLocations[0] = -1;
	attribLocations[1] = -1;
	attribLocations[2] = -1;
	attribLocations[3] = -1;
	attribLocations[4] = -1;
	attribLocations[5] = -1;
	attribLocations[6] = -1;
	attribLocations[7] = -1;
}

GLProgram::~GLProgram()
{
	clear();
}

void GLProgram::setWGS(GLint x, GLint y, GLint z)
{
	int id = getShaderID(GL_COMPUTE_SHADER);

	std::stringstream ss;
	ss << std::string("#define WGSX ") << x << std::string("\n");
	ss << std::string("#define WGSY ") << y << std::string("\n");
	ss << std::string("#define WGSZ ") << z << std::string("\n");
	ss << std::string("\n");

	this->defines[id] += ss.str();
}

void GLProgram::checkShaderStatus(GLenum shaderType)
{
	int id = getShaderID(shaderType);

	GLint gl_status;
	glGetShaderiv(shaderObjects[id], GL_COMPILE_STATUS, &gl_status);

	GLint infoLen = 0;
	glGetShaderiv(shaderObjects[id], GL_INFO_LOG_LENGTH, &infoLen);

	if(infoLen > 1)
	{
		char* infoLog = new char[infoLen];

		glGetShaderInfoLog(shaderObjects[id], infoLen, NULL, infoLog);

		if(gl_status == GL_TRUE)
		{
			LOG_WARN( LogLine() << "SHADER COMPILE_STATUS" << " - " << fileNames[id] );
		}
		else
		{
			LOG_ERR( LogLine() << "SHADER COMPILE_STATUS" << " - " << fileNames[id] );
		}

		LOG( trimString(infoLog) );

		delete[] infoLog;
	}
	else
	{
		LOG_OK( LogLine() << "SHADER COMPILE_STATUS" << " - " << fileNames[id] );
	}
}

void GLProgram::checkProgramStatus()
{
	GLint gl_status;
	glGetProgramiv(programObject, GL_LINK_STATUS, &gl_status);

	if(!gl_status)
	{
		GLint infoLen = 0;
		glGetProgramiv(programObject, GL_INFO_LOG_LENGTH, &infoLen);
		if(infoLen > 1)
		{
			char* infoLog = new char[infoLen];

			glGetProgramInfoLog(programObject, infoLen, NULL, infoLog);

			LOG_ERR( "PROGRAM LINK_STATUS" );
			LOG( trimString(infoLog) );

			delete[] infoLog;
		}
	}
	else
	{
		LOG_OK( LogLine() << "PROGRAM LINK_STATUS" );
		LOG("");
	}
}

void GLProgram::addHeader(const std::string& header)
{
	this->header = header;
}

void GLProgram::addDefine(const std::string& define, GLenum shaderType)
{
	int id = getShaderID(shaderType);
	this->defines[id] += std::string("#define ") + define + std::string("\n");
}

void GLProgram::addFile(const std::string& file, GLenum shaderType)
{
	std::string folderStr = FileSystem::GetShadersGLFolder() + file;

	if(shaderType == GL_COMPUTE_SHADER)
	{
		folderStr = FileSystem::GetKernelsGLFolder() + file;
	}

	std::fstream shaderFile( folderStr.c_str(), std::ios::in);
	bool shaderFileExists = shaderFile.is_open();

	if(shaderFileExists)
	{
		int id = getShaderID(shaderType);
		fileNames[id]= file;

		std::stringstream buffer;
		buffer << shaderFile.rdbuf();
		sources[id] += buffer.str() + "\n";
	}

	LOG_BOOL( shaderFileExists, LogLine() << "SHADER FILEOPEN" << " - " << file );
}

void GLProgram::buildProgram()
{
	for(unsigned int i = 0; i < 6; ++i)
	{
		if(sources[i].empty() == false)
		{
			GLenum type = getShaderType(i);

			shaderObjects[i] = glCreateShader(type);

			std::string source;

			if( header.empty() && (type == GL_COMPUTE_SHADER) )
			{
				if( isGLES )
				{
					header = "#version 310 es\n";
					header += "\n";
					header += "precision mediump float;\n";
					//header += "precision highp int;\n";
					//header += "precision highp uint;\n";
					header += "precision mediump sampler2D;\n";
					header += "precision mediump sampler3D;\n";
					header += "precision mediump image2D;\n";
					header += "precision mediump image3D;\n";
					header += "precision mediump iimage2D;\n";
					header += "precision mediump iimage3D;\n";
					header += "precision mediump uimage2D;\n";
					header += "precision mediump uimage3D;\n";
					header += "\n";
				}
				else
				{
					header = "#version 430 core\n";
					header += "\n";
					//header += "precision highp float;\n";
					//header += "precision highp int;\n";
					//header += "precision highp uint;\n";
					//header += "precision highp sampler2D;\n";
					//header += "precision highp sampler3D;\n";
					//header += "precision highp image2D;\n";
					//header += "precision highp image3D;\n";
					//header += "precision highp iimage2D;\n";
					//header += "precision highp iimage3D;\n";
					//header += "precision highp uimage2D;\n";
					//header += "precision highp uimage3D;\n";
					//header += "\n";
				}
			}

			if(header.empty() == false)
				source += header + "\n";

			if(defines[i].empty() == false)
				source += defines[i] + "\n";

			source += sources[i];
			const char* cc = source.c_str();

			glShaderSource(shaderObjects[i], 1, &cc, NULL);
			glCompileShader(shaderObjects[i]);

			checkShaderStatus(type);
			CHECK_GL;
		}
	}

	programObject = glCreateProgram();
	CHECK_GL;

	for(unsigned int i = 0; i < 6; ++i)
	{
		if(shaderObjects[i])
		{
			glAttachShader(programObject, shaderObjects[i]);
			CHECK_GL;
		}
	}

	glLinkProgram(programObject);
	CHECK_GL;

	checkProgramStatus();
	CHECK_GL;

	useProgram();

	if(shaderObjects[0] != 0)
	{
		attribLocations[0] = getAttribLocation("a_Position0");
		attribLocations[1] = getAttribLocation("a_Position1");
		attribLocations[2] = getAttribLocation("a_Normal0");
		attribLocations[3] = getAttribLocation("a_Normal1");
		attribLocations[4] = getAttribLocation("a_Color0");
		attribLocations[5] = getAttribLocation("a_Color1");
		attribLocations[6] = getAttribLocation("a_TexCoord0");
		attribLocations[7] = getAttribLocation("a_TexCoord1");
	}

	glUseProgram(0);

	CHECK_GL;
}

void GLProgram::useProgram()
{
	glUseProgram(programObject);
}

void GLProgram::clear()
{
	if(programObject)
	{
		glUseProgram(0);
		CHECK_GL;

		for(unsigned int i = 0; i < 6; ++i)
		{
			if(shaderObjects[i])
			{
				glDetachShader(programObject, shaderObjects[i]);
				glDeleteShader(shaderObjects[i]);
				CHECK_GL;

				shaderObjects[i] = 0;
			}

			fileNames[i].clear();

			defines[i].clear();
			sources[i].clear();
		}

		glDeleteProgram(programObject);
		CHECK_GL;

		programObject = 0;
	}

	header.clear();

	attribLocations[0] = -1;
	attribLocations[1] = -1;
	attribLocations[2] = -1;
	attribLocations[3] = -1;
	attribLocations[4] = -1;
	attribLocations[5] = -1;
	attribLocations[6] = -1;
	attribLocations[7] = -1;
}

GLint GLProgram::getAttribLocation(const std::string& attribStr) const
{
	return glGetAttribLocation(programObject, attribStr.c_str());
}

GLint GLProgram::getUniformLocation(const std::string& uniformStr) const
{
	return glGetUniformLocation(programObject, uniformStr.c_str());
}

void GLProgram::setIntValue(int value, const char* parameter) const
{
	GLint loc = glGetUniformLocation(programObject, parameter);

	if(loc == -1)
	{
		LOG_WARN( LogLine() << parameter << " is not an active uniform variable location" );
	}
	else
	{
		glUniform1i(loc, value);
	}
}

void GLProgram::setFloatValue(float value, const char* parameter) const
{
	GLint loc = glGetUniformLocation(programObject, parameter);

	if(loc == -1)
	{
		LOG_WARN( LogLine() << parameter << " is not an active uniform variable location" );
	}
	else
	{
		glUniform1f(loc, value);
	}
}

void GLProgram::setFloatVector2(const float* vec2, const char* parameter) const
{
	GLint loc = glGetUniformLocation(programObject, parameter);

	if(loc == -1)
	{
		LOG_WARN( LogLine() << parameter << " is not an active uniform variable location" );
	}
	else
	{
		glUniform2fv(loc, 1, vec2);
	}
}

void GLProgram::setFloatVector4(const float* vec4, const char* parameter) const
{
	GLint loc = glGetUniformLocation(programObject, parameter);

	if(loc == -1)
	{
		LOG_WARN( LogLine() << parameter << " is not an active uniform variable location" );
	}
	else
	{
		glUniform4fv(loc, 1, vec4);
	}
}

void GLProgram::setFloatMatrix44(const float* mat44, const char* parameter) const
{
	GLint loc = glGetUniformLocation(programObject, parameter);

	if(loc == -1)
	{
		LOG_WARN( LogLine() << parameter << " is not an active uniform variable location" );
	}
	else
	{
		glUniformMatrix4fv(loc, 1, GL_FALSE, mat44);
	}
}

void GLProgram::printActiveUniforms()
{
	GLint size, maxLength, nUniforms;
	GLsizei written;
	GLenum type;

	glGetProgramiv(programObject, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxLength);
	glGetProgramiv(programObject, GL_ACTIVE_UNIFORMS, &nUniforms);

	GLchar* name = (GLchar*)malloc(maxLength);

	printf(" Location | Name\n");
	printf("------------------------------------------------\n");
	for(int i = 0; i < nUniforms; ++i)
	{
		glGetActiveUniform(programObject, i, maxLength, &written, &size, &type, name);
		GLint location = glGetUniformLocation(programObject, name);
		printf(" %-8d | %s\n",location, name);
	}

	free(name);
}

void GLProgram::printActiveAttribs()
{
	GLint size, maxLength, nAttribs;
	GLsizei written;
	GLenum type;

	glGetProgramiv(programObject, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &maxLength);
	glGetProgramiv(programObject, GL_ACTIVE_ATTRIBUTES, &nAttribs);

	GLchar* name = (GLchar*)malloc(maxLength);

	printf(" Index | Name\n");
	printf("------------------------------------------------\n");
	for(int i = 0; i < nAttribs; ++i)
	{
		glGetActiveAttrib(programObject, i, maxLength, &written, &size, &type, name);
		GLint location = glGetAttribLocation(programObject, name);
		printf(" %-5d | %s\n",location, name);
	}

	free(name);
}