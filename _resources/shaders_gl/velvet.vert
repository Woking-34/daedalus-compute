// uniforms - used by the vertex shader
uniform mat4 u_MVPMat;
uniform mat4 u_MVMat;
uniform mat4 u_MMat;

// attributes - input to the vertex shader
attribute vec4 a_Position0;
attribute vec4 a_Normal0;

// varyings – input to the fragment shader
varying vec3 vPeye;
varying vec3 vNeye;

void main()
{
	// in most cases the matrices used are orthonormal and therefore can be used
	// to transform the vertex position and the normal
	
	vPeye = vec3(u_MVMat * a_Position0);
	vNeye = vec3(u_MVMat * a_Normal0);
	
	gl_Position = u_MVPMat * a_Position0;
}