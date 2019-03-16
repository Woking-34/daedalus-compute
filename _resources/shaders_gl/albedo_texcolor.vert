// uniforms - used by the vertex shader
uniform mat4 u_MVPMat;

// attributes - input to the vertex shader
attribute vec4 a_Position0;
attribute vec2 a_TexCoord0;

// varying variables – input to the fragment shader
varying vec2 v_TexCoord;

void main()
{
	v_TexCoord = a_TexCoord0;

    gl_Position = u_MVPMat * a_Position0;
}