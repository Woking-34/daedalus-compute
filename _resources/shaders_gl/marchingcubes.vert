#version 430

// attributes input to the vertex shader
in vec4 a_Position0;

void main(void)
{
	gl_Position = a_Position0;
}