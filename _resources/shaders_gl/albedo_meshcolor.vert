// uniforms - used by the vertex shader
uniform mat4 u_MVPMat;

// attributes - input to the vertex shader
attribute vec4 a_Position0;
 
void main()
{
    gl_Position = u_MVPMat * a_Position0;
}