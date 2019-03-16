#ifndef GLFRAMEBUFFER_H
#define GLFRAMEBUFFER_H

#include "glbase.h"

/*
The FBO is not a buffer on its own. It is a container object, much like vertex array objects.
When a FBO is bound, all drawing will go to the attached buffers of the FBO instead of the visible screen.
There are two different types of buffers that can be attached; textures and render buffers.
The texture buffer is used when the result of the operation shall be used as a texture in another rendering stage.
A render buffer, on the other hand, can't be used by a shader.

http://ephenationopengl.blogspot.hu/2012/01/setting-up-deferred-shader.html

*/

class GLFrameBuffer
{
public:
	GLFrameBuffer();
	~GLFrameBuffer();

	void createTex2D(int width, int height, bool onlyDepth = false);
	void createCubeMap(int width, int height, bool onlyDepth = false);

	void bind();
	void unbind();

	void attachCubeMap(int i);

	GLuint getID()
	{
		return fboID;
	}

	GLuint getTexID()
	{
		return texAttachID;
	}

	int getWidth()
	{
		return width;
	}

	int getHeight()
	{
		return height;
	}

protected:
	bool onlyDepth;
	int width, height;

	GLuint fboID, texAttachID, rboAttachID;
};

#endif