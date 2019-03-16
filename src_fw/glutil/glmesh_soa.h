#ifndef GLMESH__SOA_H
#define GLMESH__SOA_H

#include "glbase.h"

class MeshFile;

class GLMesh_SOA
{
public:
	GLMesh_SOA();
	~GLMesh_SOA();

	void clear();

	void createIBO(unsigned int count, void* ptr);

	void createVBO(unsigned int index, unsigned int count, void* ptr = NULL);
	void updateVBO(unsigned int index, void* ptr);

	void render(const GLint* attribLocations);

protected:
	unsigned int numVertices;
	unsigned int numPrimitives;

	GLuint meshVBID[NUMATTRIBS];
	GLuint meshIBID;

	GLenum primitiveType;
	GLsizei numVertsPrim;
};

#endif