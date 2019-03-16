// uniforms - used by the vertex shader
uniform mat4 u_MVPMat;
uniform mat4 u_MMat;

uniform vec4 u_EyeW;

// attributes - input to the vertex shader
attribute vec4 a_Position0;
attribute vec4 a_Normal0;

// varyings – input to the fragment shader
varying vec4 v_ViewW;
varying vec4 v_NormW;

void main()
{
	// in most cases the matrices used are orthonormal and therefore can be used
	// to transform the vertex position and the normal
	
	v_ViewW = u_MMat * a_Position0 - u_EyeW;
	v_NormW = u_MMat * a_Normal0;
	
	gl_Position = u_MVPMat * a_Position0;
}