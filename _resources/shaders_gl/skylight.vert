#ifdef HAVE_EGL
	precision highp float;
#endif

// uniforms used by the vertex shader
uniform mat4 VPMatrixInv;

// attributes input to the vertex shader
attribute vec4 a_Position0;

// varying variables – input to the fragment shader
varying vec4 v_PositionW;
 
void main()
{
    gl_Position = a_Position0;

	vec4 posW = VPMatrixInv * a_Position0;
    v_PositionW = posW / posW.w;
}
