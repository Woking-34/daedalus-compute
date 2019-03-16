#version 430

// attributes input to the vertex shader
attribute vec4 a_Position0;
attribute vec4 a_Normal0;

// input to the geometry shader
out vec4 v_Normal_VS2GS;

void main(void)
{
	v_Normal_VS2GS = a_Normal0;
	
	gl_Position = a_Position0;
}