#version 430

// input from geometry shader
in vec4 v_Color_GS2FS;

void main(void)
{
	gl_FragColor = v_Color_GS2FS;
}