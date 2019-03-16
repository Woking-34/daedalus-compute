#ifdef HAVE_EGL
	precision mediump float;
#endif

// input from vertex shader
varying vec4 v_Color;

void main()
{
	vec3 normal;

#ifdef HAVE_EGL
	normal.xy = gl_PointCoord * 2.0 - vec2(1.0);
#else
	normal.xy = gl_TexCoord[0].xy * 2.0 - vec2(1.0);
#endif

	float mag = dot(normal.xy, normal.xy);
	if(mag > 1.0) discard;

	normal.z = sqrt(1.0 - mag);

	float diffuse = max(0.0, dot(vec3(0.0, 0.0,1.0), normal));

	gl_FragColor = v_Color* diffuse;
}