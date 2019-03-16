uniform mat4 u_MVPMatrix;
uniform mat4 u_MMatrix;
uniform mat4 u_VPMatrix;
uniform mat4 u_PtPMatrix;
uniform sampler2D texDisp;

attribute vec4 a_Position0;

varying vec4 v_posW;
varying vec2 v_TexCoord;

void main()
{
	// ss grid
	v_posW = u_MMatrix * a_Position0;
	v_posW /= v_posW.w;
	v_posW = u_PtPMatrix * v_posW;
	v_posW /= v_posW.w;

	v_TexCoord = vec2(v_posW.x,v_posW.z) / 2000.0;
	vec4 offset = texture2DLod(texDisp, v_TexCoord, 0.0).yzxw;
	//offset = vec4(0.0,0.0,0.0,0.0);

	v_posW = v_posW + offset;
	gl_Position =  u_VPMatrix * v_posW;
}
