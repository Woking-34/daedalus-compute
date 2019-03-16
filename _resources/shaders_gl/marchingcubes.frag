#version 430

// input from geometry shader
in vec4 v_Position_GS2FS;

uniform sampler3D volumeTex;

uniform float volumeW;
uniform float volumeH;
uniform float volumeD;

void main(void)
{
	vec4 grad = vec4(
		texture(volumeTex, (v_Position_GS2FS.xyz+vec3(1.0/volumeW, 0.0, 0.0))).x - texture(volumeTex, (v_Position_GS2FS.xyz+vec3(-1.0/volumeW, 0.0, 0.0))).x, 
		texture(volumeTex, (v_Position_GS2FS.xyz+vec3(0.0, 1.0/volumeH, 0.0))).x - texture(volumeTex, (v_Position_GS2FS.xyz+vec3(0.0, -1.0/volumeH, 0.0))).x, 
		texture(volumeTex, (v_Position_GS2FS.xyz+vec3(0.0, 0.0, 1.0/volumeD))).x - texture(volumeTex, (v_Position_GS2FS.xyz+vec3(0.0, 0.0, -1.0/volumeD))).x,
		0.0
	);
	
	grad = normalize(grad);
	
	gl_FragColor = abs(grad);
}