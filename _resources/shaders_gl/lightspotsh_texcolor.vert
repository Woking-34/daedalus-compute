// uniforms - used by the vertex shader
uniform mat4 u_MVPMat;
uniform mat4 u_MMat;
uniform mat4 u_LMat;

// attributes - input to the vertex shader
attribute vec4 a_Position0;
attribute vec4 a_Normal0;
attribute vec2 a_TexCoord0;

// varying variables – input to the fragment shader
varying vec4 v_PosW;
varying vec4 v_posL;
varying vec4 v_NormW;
varying vec2 v_TexCoord;

void main()
{
	// in most cases the matrices used are orthonormal and therefore can be used
	// to transform the vertex position and the normal
	
	v_PosW = u_MMat * a_Position0;
	v_posL = u_LMat * a_Position0;
	v_NormW = u_MMat * a_Normal0;
	v_TexCoord = a_TexCoord0;
	
	gl_Position = u_MVPMat * a_Position0;
}