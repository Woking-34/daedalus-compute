// uniforms - used by the vertex shader
uniform mat4 MVPMat;

// attributes - input to the vertex shader
attribute vec4 a_Position0;
attribute vec4 a_instPos;
attribute vec4 a_instCol;

// varyings – input to the fragment shader
varying vec4 v_Color;
 
void main()
{
	v_Color = a_instCol;
	
	gl_Position = MVPMat * (a_Position0 + a_instPos);
}