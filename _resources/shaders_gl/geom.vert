#version 430

// attributes input to the vertex shader
attribute vec4 a_Position0;
attribute vec4 a_Color0;

// input to the geometry shader
out vec4 v_Color_VS2GS;

void main(void)
{
	v_Color_VS2GS = a_Color0;
	
	gl_Position = a_Position0;
}