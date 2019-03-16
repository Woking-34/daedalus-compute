#include "glmesh_soa.h"

#include "math/mathbase.h"
#include "assets/meshfile.h"

GLMesh_SOA::GLMesh_SOA()
{
	numVertices = numPrimitives = 0;

	primitiveType = numVertsPrim = 0;

	meshIBID = 0;

	meshVBID[0] = 0;
	meshVBID[1] = 0;
	meshVBID[2] = 0;
	meshVBID[3] = 0;
	meshVBID[4] = 0;
	meshVBID[5] = 0;
	meshVBID[6] = 0;
	meshVBID[7] = 0;
}

GLMesh_SOA::~GLMesh_SOA()
{
	clear();
}

void GLMesh_SOA::clear()
{
	for(int i = 0; i < NUMATTRIBS; ++i)
	{
		if(meshVBID[i])
		{
			glDeleteBuffers(1, &(meshVBID[i]));
			meshVBID[i] = 0;
		}
	}

	if(meshIBID)
	{
		glDeleteBuffers(1, &meshIBID);
		meshIBID = 0;
	}

	numVertices = numPrimitives = 0;

	primitiveType = numVertsPrim = 0;
}

void GLMesh_SOA::createIBO(unsigned int count, void* ptr)
{
	numPrimitives = count;

	primitiveType = GL_TRIANGLES;
	numVertsPrim = 3;

	glGenBuffers(1, &meshIBID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshIBID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*numPrimitives*numVertsPrim, ptr, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void GLMesh_SOA::createVBO(unsigned int index, unsigned int count, void* ptr)
{
	numVertices = count;
	if(numPrimitives == 0)
	{
		numPrimitives = count/3;
	}

	primitiveType = GL_TRIANGLES;
	numVertsPrim = 3;

	glGenBuffers(1, &(meshVBID[index]));
	glBindBuffer(GL_ARRAY_BUFFER, meshVBID[index]);
	glBufferData(GL_ARRAY_BUFFER, attribComponents[index][0]*attribComponents[index][2]*count, ptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLMesh_SOA::updateVBO(unsigned int index, void* ptr)
{
	glBindBuffer(GL_ARRAY_BUFFER, meshVBID[index]);
	glBufferSubData(GL_ARRAY_BUFFER, 0, attribComponents[index][0]*attribComponents[index][2]*numVertices, ptr);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLMesh_SOA::render(const GLint* attribLocations)
{
	if(meshIBID)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshIBID);

	for(int currAttrib = 0; currAttrib < NUMATTRIBS; ++currAttrib)
	{
		if( attribLocations[currAttrib] != -1 && meshVBID[currAttrib] )
		{
			glBindBuffer(GL_ARRAY_BUFFER, meshVBID[currAttrib]);

			glVertexAttribPointer( attribLocations[currAttrib], attribComponents[currAttrib][0], attribComponents[currAttrib][1], GL_FALSE, 0, NULL );
			glEnableVertexAttribArray( attribLocations[currAttrib] );
		}
	}
		
	if(meshIBID)
		glDrawElements(primitiveType, numPrimitives*numVertsPrim, GL_UNSIGNED_INT, NULL);
	else
		glDrawArrays(primitiveType, 0, numPrimitives*numVertsPrim);

	for(int currAttrib = 0; currAttrib < NUMATTRIBS; ++currAttrib)
	{
		if( attribLocations[currAttrib] != -1 && meshVBID[currAttrib] )
		{
			glDisableVertexAttribArray( attribLocations[currAttrib] );
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if(meshIBID)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}