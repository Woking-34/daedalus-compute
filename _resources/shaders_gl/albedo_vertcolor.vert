// uniforms - used by the vertex shader
uniform mat4 u_MVPMat;

// attributes - input to the vertex shader
attribute vec4 a_Position0;
attribute vec4 a_Color0;

// varyings – input to the fragment shader
varying vec4 v_Color;
 
void main()
{
	v_Color = a_Color0;
	
	gl_Position = u_MVPMat * a_Position0;
}