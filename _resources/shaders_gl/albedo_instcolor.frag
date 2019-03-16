// input from vertex shader
varying vec4 v_Color;

void main(void)
{
	gl_FragColor = vec4(v_Color.xyz, 1.0);
}