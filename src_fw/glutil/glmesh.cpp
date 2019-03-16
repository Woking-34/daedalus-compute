#include "glmesh.h"

#include "math/mathbase.h"
#include "assets/meshfile.h"

GLMesh::GLMesh()
{
	vertices = nullptr;
	indices = nullptr;

	meshVBID = meshIBID = 0;

	numVertices = numPrimitives = 0;

	primitiveType = numVertsPrim = 0;
	
	stride = 0;

	attribOffsets[0] = -1;
	attribOffsets[1] = -1;
	attribOffsets[2] = -1;
	attribOffsets[3] = -1;
	attribOffsets[4] = -1;
	attribOffsets[5] = -1;
	attribOffsets[6] = -1;
	attribOffsets[7] = -1;
}

GLMesh::~GLMesh()
{
	clear();
}

void GLMesh::clear()
{
	if(vertices)
	{
		::operator delete(vertices);
		vertices = nullptr;
	}

	if(indices)
	{
		delete[] indices;
		indices = nullptr;
	}

	if(meshVBID)
	{
		glDeleteBuffers(1, &meshVBID);
		CHECK_GL;

		meshVBID = 0;
	}

	if(meshIBID)
	{
		glDeleteBuffers(1, &meshIBID);
		CHECK_GL;

		meshIBID = 0;
	}

	numVertices = numPrimitives = 0;

	primitiveType = numVertsPrim = 0;

	stride = 0;

	attribOffsets[0] = -1;
	attribOffsets[1] = -1;
	attribOffsets[2] = -1;
	attribOffsets[3] = -1;
	attribOffsets[4] = -1;
	attribOffsets[5] = -1;
	attribOffsets[6] = -1;
	attribOffsets[7] = -1;
}

void GLMesh::create(const MeshFile& meshSource)
{
	stride = 0;
	for(int currAttrib = 0; currAttrib < NUMATTRIBS; ++currAttrib)
	{
		if( meshSource.hasAttribute[currAttrib] )
		{
			attribOffsets[currAttrib] = stride;
			stride += attribComponents[currAttrib][0] * attribComponents[currAttrib][2];
		}
	}

	numVertices = meshSource.numVertices;
	numPrimitives = meshSource.numPrimitives;

	vertices = ::operator new(stride*numVertices);

	if(meshSource.indicesVecFlat.size())
		indices = new GLuint[meshSource.numPrimitives * meshSource.numVertsPrim];

	for(unsigned int currVertexI = 0; currVertexI < numVertices; ++currVertexI)
	{
		unsigned int currAttrib;

		// vertices vec0 - attrib0
		currAttrib = 0;
		if( meshSource.hasAttribute[currAttrib] )
		{
			*((Vec4f*)((unsigned char*)vertices + currVertexI*stride + attribOffsets[currAttrib])) = meshSource.position0Vec[currVertexI];
		}

		// normals vec0 - attrib2
		currAttrib = 2;
		if( meshSource.hasAttribute[currAttrib] )
		{
			*((Vec4f*)((unsigned char*)vertices + currVertexI*stride + attribOffsets[currAttrib])) = meshSource.normals0Vec[currVertexI];
		}

		// colors vec0 - attrib4
		currAttrib = 4;
		if( meshSource.hasAttribute[currAttrib] )
		{
			*((Vec4f*)((unsigned char*)vertices + currVertexI*stride + attribOffsets[currAttrib])) = meshSource.colors0Vec[currVertexI];
		}

		// uvcoord vec0 - attrib6
		currAttrib = 6;
		if( meshSource.hasAttribute[currAttrib] )
		{
			*((Vec2f*)((unsigned char*)vertices + currVertexI*stride + attribOffsets[currAttrib])) = meshSource.uvcoord0Vec[currVertexI];
		}
	}

	if(meshSource.primType == MESHPRIM_TYPE_POINTS)
	{
		primitiveType = GL_POINTS;
		numVertsPrim = 1;
	}
	else if(meshSource.primType == MESHPRIM_TYPE_LINES)
	{
		primitiveType = GL_LINES;
		numVertsPrim = 2;

		if(meshSource.indicesVecFlat.size())
		{
			for(unsigned int currIndexI = 0; currIndexI < numPrimitives; ++currIndexI)
			{
				indices[ currIndexI*meshSource.numVertsPrim + 0 ] = meshSource.indicesVecFlat[ currIndexI*meshSource.numVertsPrim + 0 ];
				indices[ currIndexI*meshSource.numVertsPrim + 1 ] = meshSource.indicesVecFlat[ currIndexI*meshSource.numVertsPrim + 1 ];
			}
		}
	}
	else if(meshSource.primType == MESHPRIM_TYPE_TRIS)
	{
		primitiveType = GL_TRIANGLES;
		numVertsPrim = 3;

		if(meshSource.indicesVecFlat.size())
		{
			for(unsigned int currIndexI = 0; currIndexI < numPrimitives; ++currIndexI)
			{
				indices[ currIndexI*meshSource.numVertsPrim + 0 ] = meshSource.indicesVecFlat[ currIndexI*meshSource.numVertsPrim + 0 ];
				indices[ currIndexI*meshSource.numVertsPrim + 1 ] = meshSource.indicesVecFlat[ currIndexI*meshSource.numVertsPrim + 1 ];
				indices[ currIndexI*meshSource.numVertsPrim + 2 ] = meshSource.indicesVecFlat[ currIndexI*meshSource.numVertsPrim + 2 ];
			}
		}
	}
	else if(meshSource.primType == MESHPRIM_TYPE_QUADS)
	{
		primitiveType = GL_QUADS;
		numVertsPrim = 4;

		if(meshSource.indicesVecFlat.size())
		{
			for(unsigned int currIndexI = 0; currIndexI < numPrimitives; ++currIndexI)
			{
				indices[ currIndexI*meshSource.numVertsPrim + 0 ] = meshSource.indicesVecFlat[ currIndexI*meshSource.numVertsPrim + 0 ];
				indices[ currIndexI*meshSource.numVertsPrim + 1 ] = meshSource.indicesVecFlat[ currIndexI*meshSource.numVertsPrim + 1 ];
				indices[ currIndexI*meshSource.numVertsPrim + 2 ] = meshSource.indicesVecFlat[ currIndexI*meshSource.numVertsPrim + 2 ];
				indices[ currIndexI*meshSource.numVertsPrim + 3 ] = meshSource.indicesVecFlat[ currIndexI*meshSource.numVertsPrim + 3 ];
			}
		}
	}

	glGenBuffers(1, &meshVBID);
	glBindBuffer(GL_ARRAY_BUFFER, meshVBID);
	glBufferData(GL_ARRAY_BUFFER, stride*numVertices, vertices, GL_STATIC_DRAW);

	if(meshSource.indicesVecFlat.size())
	{
		glGenBuffers(1, &meshIBID);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshIBID);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*numPrimitives*numVertsPrim, indices, GL_STATIC_DRAW);
	}
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void GLMesh::render(const GLint* attribLocations)
{
	if(meshIBID)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshIBID);

	glBindBuffer(GL_ARRAY_BUFFER, meshVBID);

	for(int currAttrib = 0; currAttrib < NUMATTRIBS; ++currAttrib)
	{
		if( attribLocations[currAttrib] != -1 )
		{
			glVertexAttribPointer( attribLocations[currAttrib], attribComponents[currAttrib][0], attribComponents[currAttrib][1], GL_FALSE, stride, ((char *)0 + (attribOffsets[currAttrib])) );
			glEnableVertexAttribArray( attribLocations[currAttrib] );
		}
	}
	
	if(meshIBID)
		glDrawElements(primitiveType, numPrimitives*numVertsPrim, GL_UNSIGNED_INT, NULL);
	else
		glDrawArrays(primitiveType, 0, numPrimitives*numVertsPrim);

	for(int currAttrib = 0; currAttrib < NUMATTRIBS; ++currAttrib)
	{
		if( attribLocations[currAttrib] != -1 )
		{
			glDisableVertexAttribArray( attribLocations[currAttrib] );
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if(meshIBID)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}