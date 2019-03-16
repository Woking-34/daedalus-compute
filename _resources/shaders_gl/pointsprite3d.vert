// uniforms used by the vertex shader
uniform mat4 u_MVMatrix;
uniform mat4 u_MVPMatrix;
uniform float u_pSize;
uniform float u_pScale;

// attributes input to the vertex shader
attribute vec4 a_Position0;
attribute vec4 a_Color0;

// varyings – input to the fragment shader
varying vec4 v_Color;

void main()
{
	vec4 posEye = u_MVMatrix * a_Position0;
	float dist = length(posEye.xyz);
	
	v_Color = a_Color0;
	
	gl_PointSize = u_pSize * (u_pScale / dist);
	
	gl_Position = u_MVPMatrix * a_Position0;
}