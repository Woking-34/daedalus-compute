// uniforms used by the fragment shader
uniform samplerCube envMap;

// input from vertex shader
varying vec4 v_ViewW;
varying vec4 v_NormW;

void main(void)
{
	vec3 reflectDir = reflect( normalize(v_ViewW.xyz), normalize(v_NormW.xyz) );
	gl_FragColor = textureCube(envMap, reflectDir);

	//gl_FragColor = textureCube(envMap, normalize(v_NormW.xyz));
}