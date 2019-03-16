#ifndef GLMESH_H
#define GLMESH_H

#include "glbase.h"

class MeshFile;

class GLMesh
{
public:
	GLMesh();
	~GLMesh();

	void create(const MeshFile& meshSource);
	void render(const GLint* attribLocations);

	void clear();

protected:
	unsigned int numVertices;
	unsigned int numPrimitives;

	GLuint meshVBID;
	GLuint meshIBID;

	GLenum primitiveType;
	GLsizei numVertsPrim;

	GLsizei stride;

	GLsizei attribOffsets[NUMATTRIBS];

	GLvoid* vertices;
	GLuint* indices;
};

#endif