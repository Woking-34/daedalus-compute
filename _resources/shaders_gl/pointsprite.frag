//precision mediump float;

varying vec4 v_Color;

void main()
{
	vec3 normal;
	normal.xy = gl_PointCoord * 2.0 - vec2(1.0);
	float mag = dot(normal.xy, normal.xy);
	if (mag > 1.0) discard;
	normal.z = sqrt(1.0 - mag);

	float diffuse = max(0.0, dot(vec3(0.0, 0.0, 1.0), normal));
	
	gl_FragColor = v_Color * diffuse;
}