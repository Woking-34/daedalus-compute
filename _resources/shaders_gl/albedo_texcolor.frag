// uniforms - used by the fragment shader
uniform sampler2D albedoTex;

// input from vertex shader
varying vec2 v_TexCoord;

void main(void)
{
	gl_FragColor = texture2D(albedoTex, v_TexCoord);
}