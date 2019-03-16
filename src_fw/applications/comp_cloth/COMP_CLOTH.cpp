#include "COMP_CLOTH.h"


COMP_CLOTH::COMP_CLOTH() : bufferPositions(nullptr), bufferNormals(nullptr), bufferWeights(nullptr)
{
	vboInCurr = 0;
	vboInPrev = 0;
	vboOutCurr = 0;
	vboOutPrev = 0;
}

COMP_CLOTH::~COMP_CLOTH()
{

}