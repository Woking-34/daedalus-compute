#include "COMPGL_PARTICLES.h"

#include "system/log.h"
#include "system/filesystem.h"

COMPGL_PARTICLES::COMPGL_PARTICLES()
{
	vel0SSBO = 0;
	vel1SSBO = 0;

	hListSSBO = 0;
	pListSSBO = 0;
}

COMPGL_PARTICLES::~COMPGL_PARTICLES()
{

}

void COMPGL_PARTICLES::init()
{
	{
		computeProgram_reset.setWGS(wgsX, wgsY, wgsZ);
		
		computeProgram_reset.addFile("particles/particles_reset.comp", GL_COMPUTE_SHADER);
		computeProgram_reset.buildProgram();
	}

	{
		computeProgram_create.setWGS(wgsX, wgsY, wgsZ);
		
		computeProgram_create.addFile("particles/particles_create.comp", GL_COMPUTE_SHADER);
		computeProgram_create.buildProgram();
	}

	{
		computeProgram_collide.setWGS(wgsX, wgsY, wgsZ);
		
		computeProgram_collide.addFile("particles/particles_collide.comp", GL_COMPUTE_SHADER);
		computeProgram_collide.buildProgram();
	}

	{
		computeProgram_integrate.setWGS(wgsX, wgsY, wgsZ);
		
		computeProgram_integrate.addFile("particles/particles_integrate.comp", GL_COMPUTE_SHADER);
		computeProgram_integrate.buildProgram();
	}

	{
		glGenBuffers(1, &vel0SSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, vel0SSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, numParticles * 4 * sizeof(float), bufferVel, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	{
		glGenBuffers(1, &vel1SSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, vel1SSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, numParticles * 4 * sizeof(float), bufferVel, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	{
		glGenBuffers(1, &hListSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, hListSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, numGridCellsPaddded*sizeof(int), NULL, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	{
		glGenBuffers(1, &pListSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, pListSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, numParticles*sizeof(int), NULL, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();

	CHECK_GL;
}

void COMPGL_PARTICLES::terminate()
{
	if (vel0SSBO)
	{
		glDeleteBuffers(1, &vel0SSBO);
		vel0SSBO = 0;
	}
	if (vel1SSBO)
	{
		glDeleteBuffers(1, &vel1SSBO);
		vel1SSBO = 0;
	}

	if (hListSSBO)
	{
		glDeleteBuffers(1, &hListSSBO);
		hListSSBO = 0;
	}
	if (pListSSBO)
	{
		glDeleteBuffers(1, &pListSSBO);
		pListSSBO = 0;
	}
}

void COMPGL_PARTICLES::compute()
{
	CHECK_GL;

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, hListSSBO);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pListSSBO);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vboPos);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, vel0SSBO);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, vel1SSBO);

	for(int i = 0; i < 5; ++i)
	{
		{
			computeProgram_reset.useProgram();

			glDispatchCompute(numGridCellsPaddded/wgsX, 1, 1);
			glUseProgram(0);

			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}

		{
			computeProgram_create.useProgram();

			computeProgram_create.setFloatValue(particleRadius, "particleRadius");
			computeProgram_create.setIntValue(gridCells, "gridSize");

			glDispatchCompute(numParticles/wgsX, 1, 1);
			glUseProgram(0);

			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}

		{
			computeProgram_collide.useProgram();

			computeProgram_collide.setFloatValue(particleRadius, "particleRadius");
			computeProgram_collide.setIntValue(gridCells, "gridSize");

			glDispatchCompute(numParticles/wgsX, 1, 1);
			glUseProgram(0);

			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}

		{
			computeProgram_integrate.useProgram();

			computeProgram_integrate.setFloatValue(particleRadius, "particleRadius");

			glDispatchCompute(numParticles/wgsX, 1, 1);
			glUseProgram(0);

			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}
	}

	glFinish();

	CHECK_GL;
}

void COMPGL_PARTICLES::download()
{
	CHECK_GL;

	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, vboInCurr);
	//	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, launchW*launchH*4*sizeof(float), bufferPositions);
	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, vboPos);

		void* glMem = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, numParticles * 4 * sizeof(float), GL_MAP_READ_BIT);

		if (glMem)
			memcpy(bufferPos, glMem, numParticles * 4 * sizeof(float));

		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	}

	glFinish();

	CHECK_GL;
}