#include "COMPGL_CLOTH.h"

#include "system/log.h"
#include "system/filesystem.h"

COMPGL_CLOTH::COMPGL_CLOTH()
{
	massSSBO = 0;
}

COMPGL_CLOTH::~COMPGL_CLOTH()
{

}

void COMPGL_CLOTH::init()
{
	clothProgram.setWGS(wgsX, wgsY, 1);	
	clothProgram.addFile("cloth/cloth.comp", GL_COMPUTE_SHADER);
	clothProgram.buildProgram();
	
	{
		glGenBuffers(1, &massSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, massSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, launchW*launchH*sizeof(float), bufferWeights, GL_STATIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();

	CHECK_GL;
}

void COMPGL_CLOTH::terminate()
{
	if (massSSBO)
	{
		glDeleteBuffers(1, &massSSBO);
		massSSBO = 0;
	}
}

void COMPGL_CLOTH::compute()
{
	CHECK_GL;

	clothProgram.useProgram();

	clothProgram.setFloatValue(damp, "damp");
	clothProgram.setFloatValue(dt, "dt");
	clothProgram.setFloatValue(stepX, "stepX");
	clothProgram.setFloatValue(stepY, "stepY");

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vboInCurr);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vboInPrev);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vboOutCurr);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, vboOutPrev);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, vboNormals);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, massSSBO);

	glDispatchCompute(launchW/wgsX, launchH/wgsY, 1);

	glUseProgram(0);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	glFinish();

	CHECK_GL;

	std::swap(vboInCurr, vboOutCurr);
	std::swap(vboInPrev, vboOutPrev);
}

void COMPGL_CLOTH::download()
{
	CHECK_GL;

	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, vboInCurr);
	//	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, launchW*launchH*4*sizeof(float), bufferPositions);
	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, vboNormals);
	//	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, launchW*launchH*4*sizeof(float), bufferNormals);
	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, vboInCurr);

		void* glMem = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, launchW*launchH * 4 * sizeof(float), GL_MAP_READ_BIT);

		if (glMem)
			memcpy(bufferPositions, glMem, launchW*launchH * 4 * sizeof(float));

		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	}

	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, vboNormals);

		void* glMem = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, launchW*launchH * 4 * sizeof(float), GL_MAP_READ_BIT);

		if (glMem)
			memcpy(bufferNormals, glMem, launchW*launchH * 4 * sizeof(float));

		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	}

	glFinish();

	CHECK_GL;
}