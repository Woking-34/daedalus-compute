#version 430

#extension GL_EXT_gpu_shader4 : enable

layout(points) in;
layout(triangle_strip, max_vertices=16) out;

uniform sampler3D volumeTex;
uniform isampler2D cubeFlagsTex;
uniform isampler2D triTableTex;

uniform mat4 u_MVPMat;

uniform float isolevel;
uniform float volumeW;
uniform float volumeH;
uniform float volumeD;

out vec4 v_Position_GS2FS;

vec4 vertOff[8] = vec4[8]
(
	vec4(0.0, 0.0, 0.0, 0.0), vec4(1.0, 0.0, 0.0, 0.0), vec4(1.0, 1.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0),
	vec4(0.0, 0.0, 1.0, 0.0), vec4(1.0, 0.0, 1.0, 0.0), vec4(1.0, 1.0, 1.0, 0.0), vec4(0.0, 1.0, 1.0, 0.0)
);

int edgeCon[12][2] = int[12][2]
(
	int[2](0,1), int[2](1,2), int[2](2,3), int[2](3,0),
	int[2](4,5), int[2](5,6), int[2](6,7), int[2](7,4),
	int[2](0,4), int[2](1,5), int[2](2,6), int[2](3,7)
);

vec4 edgeDir[12] = vec4[12]
(
	vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(-1.0, 0.0, 0.0, 0.0), vec4(0.0, -1.0, 0.0, 0.0),
	vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(-1.0, 0.0, 0.0, 0.0), vec4(0.0, -1.0, 0.0, 0.0),
	vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4( 0.0, 0.0, 1.0, 0.0), vec4(0.0,  0.0, 1.0, 0.0)
);

int cubeFlags(int i)
{
	return texelFetch2D(cubeFlagsTex, ivec2(i, 0), 0).x;
}

int triTable(int i, int j)
{
	return texelFetch2D(triTableTex, ivec2(j, i), 0).x;
}

float fGetOffset(float fValue1, float fValue2, float fValueDesired)
{
	float fDelta = fValue2 - fValue1;
	if(fDelta == 0.0)
		return 0.5;
	else
		return (fValueDesired - fValue1)/fDelta;
}

void main(void)
{
	float fTargetValue = isolevel;
	float dCorners[8];
	
	for(int iCorner = 0; iCorner < 8; ++iCorner)
	{
		vec4 coord = (gl_in[0].gl_Position + vertOff[iCorner]) / vec4(volumeW,volumeH,volumeD,1.0);
		
		dCorners[iCorner] = texture(volumeTex, coord.xyz).x;
	}
	
	int iFlagIndex = 0;
	iFlagIndex += int(dCorners[0] < fTargetValue);
	iFlagIndex += int(dCorners[1] < fTargetValue)*2;
	iFlagIndex += int(dCorners[2] < fTargetValue)*4;
	iFlagIndex += int(dCorners[3] < fTargetValue)*8;
	iFlagIndex += int(dCorners[4] < fTargetValue)*16;
	iFlagIndex += int(dCorners[5] < fTargetValue)*32;
	iFlagIndex += int(dCorners[6] < fTargetValue)*64;
	iFlagIndex += int(dCorners[7] < fTargetValue)*128;
	
	int iEdgeFlags = cubeFlags(iFlagIndex);
	if(iEdgeFlags == 0 || iEdgeFlags == 255) 
		return;
	
	vec4 edgeVerts[12];
	
	for(int iEdge = 0; iEdge < 12; ++iEdge)
	{
		//if( bool(iEdgeFlags) && bool(1<<iEdge) )
		{
			float fOffset = fGetOffset( dCorners[ edgeCon[iEdge][0] ], dCorners[ edgeCon[iEdge][1] ], fTargetValue );
			
			vec4 edgeVert = gl_in[0].gl_Position + (vertOff[ edgeCon[iEdge][0] ]  +  fOffset * edgeDir[iEdge]);
			
			edgeVerts[iEdge] = edgeVert;
		}
	}
	
	for(int iTriangle = 0; iTriangle < 5; ++iTriangle)
	{
		if( triTable(iFlagIndex, 3*iTriangle) < 0 )
			return;
		
		for(int iCorner = 0; iCorner < 3; ++iCorner)
		{
			int iVertex = triTable(iFlagIndex, 3*iTriangle+iCorner);
			
			v_Position_GS2FS = edgeVerts[iVertex] / vec4(volumeW,volumeH,volumeD,1.0);
			gl_Position = u_MVPMat * v_Position_GS2FS;
			EmitVertex();
		}
		
		EndPrimitive();
	}
}