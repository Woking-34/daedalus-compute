#include "COMPGL_DEFORM.h"

#include "system/log.h"
#include "system/filesystem.h"

COMPGL_DEFORM::COMPGL_DEFORM()
{

}

COMPGL_DEFORM::~COMPGL_DEFORM()
{

}

void COMPGL_DEFORM::init()
{
	deformProgram.setWGS(static_cast<GLint>(wgsX), static_cast<GLint>(wgsY), 1);
	deformProgram.addFile("deform/deform.comp", GL_COMPUTE_SHADER);
	deformProgram.buildProgram();

	CHECK_GL;
}

void COMPGL_DEFORM::terminate()
{

}

void COMPGL_DEFORM::compute(int currDeform, float currTime)
{
	CHECK_GL;

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vbo);

	deformProgram.useProgram();
	
	deformProgram.setIntValue(currDeform, "currDeform");
	deformProgram.setFloatValue(currTime, "currTime");
	
	deformProgram.setFloatValue(sizeX, "gridWidth");
	deformProgram.setFloatValue(sizeY, "gridLength");
	deformProgram.setFloatValue(stepX, "gridWidthDt");
	deformProgram.setFloatValue(stepY, "gridLengthDt");

	glDispatchCompute(static_cast<GLuint>(launchW/wgsX), static_cast<GLuint>(launchH/wgsY), 1u);

	glUseProgram(0);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();

	CHECK_GL;
}

void COMPGL_DEFORM::download()
{
	CHECK_GL;

	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, vbo);

		void* glMem = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 2 * launchW*launchH * 4 * sizeof(float), GL_MAP_READ_BIT);

		if (glMem)
			memcpy(bufferVertices, glMem, 2 * launchW*launchH * 4 * sizeof(float));

		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	}

	glFinish();

	CHECK_GL;
}