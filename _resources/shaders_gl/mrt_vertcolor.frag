// input from vertex shader
varying vec4 v_PosW;
varying vec4 v_NormW;
varying vec4 v_Col;

void main(void)
{
	gl_FragData[0] = v_PosW;
	gl_FragData[1] = v_NormW;
	gl_FragData[2] = v_Col;
}