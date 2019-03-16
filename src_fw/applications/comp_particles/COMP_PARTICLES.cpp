#include "COMP_PARTICLES.h"


COMP_PARTICLES::COMP_PARTICLES() : bufferPos(nullptr), bufferCol(nullptr), bufferVel(nullptr)
{
	vboPos = 0;
	vboCol = 0;
}

COMP_PARTICLES::~COMP_PARTICLES()
{

}