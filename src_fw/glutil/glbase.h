#ifndef GLBASE_H
#define GLBASE_H

#include "system/log.h"

#if defined(BUILD_APPLE)
	#include <OpenGL/gl.h>
    #include <GLUT/glut.h>
#else
	#include <GL/glew.h>
	#include <GL/freeglut.h>

	#if defined(BUILD_UNIX)
		#include <GL/glx.h>
	#endif
#endif

#define CHECK_GL { GLenum glStatus = glGetError(); if( glStatus != GL_NO_ERROR ) { LOG_ERR( LogLine() << "File: " << __FILE__ << " " << "Line: " << __LINE__ << " " << "OpenGL error: " << openglGetErrorString( glStatus ) ); } }
std::string openglGetErrorString(GLenum glStatus);

#define NUMATTRIBS 8

static GLint attribComponents[NUMATTRIBS][3] =
{
	{4, GL_FLOAT, sizeof(float)},
	{4, GL_FLOAT, sizeof(float)},
	{4, GL_FLOAT, sizeof(float)},
	{4, GL_FLOAT, sizeof(float)},
	{4, GL_FLOAT, sizeof(float)},
	{4, GL_FLOAT, sizeof(float)},
	{2, GL_FLOAT, sizeof(float)},
	{2, GL_FLOAT, sizeof(float)}
};

INLINE void drawSphere(float r, int n)
{
	float x, y, z;

	for (int j=0; j<n/2; j++)
	{
		float theta1 = j * (2.0f*(float)(M_PI)) / n;
		float theta2 = (j + 1) * (2.0f*(float)(M_PI)) / n;

		glBegin(GL_QUAD_STRIP);
		for (int i=0; i<=n; i++)
		{
			float fi = i * (2.0f*(float)(M_PI)) / n;
			
			x = r*sin(theta2) * cos(fi);
			y = r*cos(theta2);
			z = r*sin(theta2) * sin(fi);
			glVertex3f(x,y,z);

			x = r*sin(theta1) * cos(fi);
			y = r*cos(theta1);
			z = r*sin(theta1) * sin(fi);
			glVertex3f(x,y,z);
		}
		glEnd();
	}
}

INLINE void drawTorus(float r, float R, int n)
{
	float x, y, z;

	for (int j=0; j<n; j++)
	{
      float theta1 = j * (2.0f*(float)(M_PI)) / n;
      float theta2 = (j + 1) * (2.0f*(float)(M_PI)) / n;

      glBegin(GL_QUAD_STRIP);
      for (int i=0; i<=n; i++)
	  {
         float fi = i * (2.0f*(float)(M_PI)) / n;

         x = (R + r*cos(theta2)) * cos(fi);
         y = r * sin(theta2);
         z = (R + r*cos(theta2)) * sin(fi);
         glVertex3f(x,y,z);

         x = (R + r*cos(theta1)) * cos(fi);
         y = r * sin(theta1);
         z = (R + r*cos(theta1)) * sin(fi);
         glVertex3f(x,y,z);
      }
      glEnd();
   }
}

#endif