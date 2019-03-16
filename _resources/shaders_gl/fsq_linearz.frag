uniform sampler2D imgTex;

uniform float near;
uniform float far;

// input from vertex shader
varying vec2 v_TexCoord;

void main(void)
{
	float z = texture2D(imgTex, v_TexCoord).x;
	float linearZ = (2.0 * near) / (far + near - z * (far - near));
	
	gl_FragColor = vec4(linearZ,linearZ,linearZ,1.0);
	//gl_FragColor = vec4(z,z,z,1.0);
}