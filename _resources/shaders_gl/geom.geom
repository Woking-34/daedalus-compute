#version 430

layout(points) in;
layout(triangle_strip, max_vertices=18) out;

uniform float u_radius;

in vec4 v_Color_VS2GS[];
out vec4 v_Color_GS2FS;

void main(void)
{
	v_Color_GS2FS = v_Color_VS2GS[0];
	
	float dphi = 6.283185307179586232 / 8.0;
	
	for(int i = 0; i <= 8; ++i)
	{
		float x = sin(dphi*i) * u_radius;
		float y = cos(dphi*i) * u_radius;
		
		gl_Position = gl_in[0].gl_Position + vec4(x,y,0.0,0.0);
		EmitVertex();
		
		gl_Position = gl_in[0].gl_Position;
		EmitVertex();
	}
	
	EndPrimitive();
}