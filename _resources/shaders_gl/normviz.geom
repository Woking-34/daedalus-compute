#version 430

layout(triangles) in;
layout(line_strip, max_vertices=6) out;

uniform mat4 u_MVPMat;
uniform float length;

in vec4 v_Normal_VS2GS[];
out vec4 v_Color_GS2FS;

void main(void)
{
	for(int i=0; i<gl_in.length(); ++i)
	{
		//int i = 0;
		vec4 P = gl_in[i].gl_Position;
		vec4 N = v_Normal_VS2GS[i];
		
		v_Color_GS2FS = abs(N);
		gl_Position = u_MVPMat * P;
		EmitVertex();
		
		gl_Position = u_MVPMat * (P + N * length);
		EmitVertex();
		
		EndPrimitive();
	}
}