// uniforms used by the fragment shader
uniform sampler3D volumeTex;
uniform vec4 eyePosW;

// input from vertex shader
varying vec4 v_PositionW;

bool intersectRayAABB( vec4 rayOrg, vec4 rayDir, vec4 aabbMin, vec4 aabbMax, out float tNear, out float tFar)
{
	vec4 dirInv = 1.0 / rayDir;
	
	vec4 tnear4 = dirInv * (aabbMin - rayOrg);
	vec4 tfar4 = dirInv * (aabbMax - rayOrg);
	
	vec4 t0 = min(tnear4, tfar4);
	vec4 t1 = max(tnear4, tfar4);
	
	tNear = max(max(t0.x, t0.y), t0.z);
	tFar = min(min(t1.x, t1.y), t1.z);
	
	return (tFar >= tNear) && (tFar > 0.0);
}

void main(void)
{
	vec4 viewDirW = normalize(v_PositionW - eyePosW);
	bool hasHit = false;
	
	vec4 aabbMin = vec4( 0.0, 0.0, 0.0, 1.0);
	vec4 aabbMax = vec4( 1.0, 1.0, 1.0, 1.0);
	float tNear, tFar;
	
	hasHit = intersectRayAABB(eyePosW, viewDirW, aabbMin, aabbMax, tNear, tFar);
	
	vec4 bgColor = vec4(0.0, 0.0, 0.0, 1.0);
	vec4 dst = vec4(0.0);
	
	if(hasHit)
	{
		//float maxDist = length( aabbMax - aabbMin );
		//float dist = tFar - tNear;
		//float color = dist / maxDist;
		//
		//gl_FragColor = vec4(color, color, color, 1.0);
		
		vec4 entryPoint = tNear * viewDirW + eyePosW;
		vec4 exitPoint = tFar * viewDirW + eyePosW;
		
		vec4 voxelCoord = entryPoint;
		
		float value = 0.0;
		
		for( int i = 0; i < 256; ++i )
		{
			value =  texture3D(volumeTex, voxelCoord.xyz).x;

			vec4 src = vec4(value);
			src.a *= 0.75;
			
			src.rgb *= src.a;
			dst += (1.0 - dst.a)*src;
						
			voxelCoord += viewDirW * 0.01;
			
			if(voxelCoord.x > 1.0 || voxelCoord.y > 1.0 || voxelCoord.z > 1.0)
				break;
				
			if(voxelCoord.x < 0.0 || voxelCoord.y < 0.0 || voxelCoord.z < 0.0)
				break;
		}
		
		gl_FragColor = dst;
		return;
	}
	
	discard;
}