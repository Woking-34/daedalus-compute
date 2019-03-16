#include "glframebuffer.h"

#include "system/log.h"

GLFrameBuffer::GLFrameBuffer()
{
	onlyDepth = false;
	fboID = texAttachID = rboAttachID = 0;
}

GLFrameBuffer::~GLFrameBuffer()
{
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	if(rboAttachID)
		glDeleteRenderbuffers(1, &rboAttachID);

	glBindTexture(GL_TEXTURE_2D, 0);
	if(texAttachID)
		glDeleteTextures(1, &texAttachID);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);	
	if(fboID)
		glDeleteFramebuffers(1, &fboID);
}

void GLFrameBuffer::createTex2D(int width, int height, bool onlyDepth)
{
	this->width = width;
	this->height = height;
	this->onlyDepth = onlyDepth;

	// create texture attachment
	glGenTextures(1, &texAttachID);
	glBindTexture(GL_TEXTURE_2D, texAttachID);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	if(!onlyDepth)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	}
	else
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	}

	glBindTexture(GL_TEXTURE_2D, 0);

	// create renderbuffer attachment
	glGenRenderbuffers(1, &rboAttachID);
	glBindRenderbuffer(GL_RENDERBUFFER, rboAttachID);

	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	// create fbo
	glGenFramebuffers(1, &fboID);
    glBindFramebuffer(GL_FRAMEBUFFER, fboID);

	if(!onlyDepth)
	{
		// attach a texture to FBO color attachement point
		//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texAttachID, 0);

		// attach a renderbuffer to FBO depth attachment point
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboAttachID);
	}
	else
	{
		// attach a texture to FBO depth attachement point
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texAttachID, 0);

		// disable color buffer if you don't attach any color buffer image
		glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
	}

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	switch(status)
    {
		case GL_FRAMEBUFFER_COMPLETE:
			LOG_OK( LogLine() << "GL_FRAMEBUFFER_COMPLETE");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
			LOG_ERR( LogLine() << "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
			LOG_ERR( LogLine() << "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
			LOG_ERR( LogLine() << "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
			LOG_ERR( LogLine() << "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER");
			break;
		default:
			LOG_ERR( LogLine() << "GL_FRAMEBUFFER_UNKNOWN_ERROR");
			break;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	LOG("");
}

void GLFrameBuffer::createCubeMap(int width, int height, bool onlyDepth)
{
	this->width = width;
	this->height = height;
	this->onlyDepth = onlyDepth;

	// create texture attachment
	glGenTextures(1, &texAttachID);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texAttachID);

	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	if(!onlyDepth)
	{
		for(int face = 0; face < 6; ++face)
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_RGBA8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
		}
	}
	else
	{
		for(int face = 0; face < 6; ++face)
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
		}
	}

	glBindTexture(GL_TEXTURE_2D, 0);

	// create renderbuffer attachment
	glGenRenderbuffers(1, &rboAttachID);
	glBindRenderbuffer(GL_RENDERBUFFER, rboAttachID);

	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	// create fbo
	glGenFramebuffers(1, &fboID);
    glBindFramebuffer(GL_FRAMEBUFFER, fboID);

	if(!onlyDepth)
	{
		// attach a texture to FBO color attachement point
		//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X, texAttachID, 0);

		// attach a renderbuffer to FBO depth attachment point
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboAttachID);
	}
	else
	{
		// attach a texture to FBO depth attachement point
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_CUBE_MAP_POSITIVE_X, texAttachID, 0);

		// disable color buffer if you don't attach any color buffer image
		glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
	}

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	switch(status)
    {
		case GL_FRAMEBUFFER_COMPLETE:
			LOG_OK( LogLine() << "GL_FRAMEBUFFER_COMPLETE");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
			LOG_ERR( LogLine() << "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
			LOG_ERR( LogLine() << "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
			LOG_ERR( LogLine() << "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
			LOG_ERR( LogLine() << "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER");
			break;
		default:
			LOG_ERR( LogLine() << "GL_FRAMEBUFFER_COMPLETE_UNKNOWN_ERROR");
			break;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	LOG("");
}

void GLFrameBuffer::bind()
{
	glBindFramebuffer(GL_FRAMEBUFFER, fboID);	
}

void GLFrameBuffer::unbind()
{
	// switch back to window-system-provided framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);	
}

void GLFrameBuffer::attachCubeMap(int i)
{
	if(!onlyDepth)
	{
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X+i, texAttachID, 0);
	}
	else
	{
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_CUBE_MAP_POSITIVE_X+i, texAttachID, 0);
	}
}