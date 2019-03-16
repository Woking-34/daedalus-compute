uniform sampler2D imgTex;

// input from vertex shader
varying vec2 v_TexCoord;

void main(void)
{
	gl_FragColor = texture2D(imgTex, v_TexCoord);
	//gl_FragColor = texture2D(imgTex, vec2(1.0, -1.0) * v_TexCoord + vec2(0.0, 1.0));
}