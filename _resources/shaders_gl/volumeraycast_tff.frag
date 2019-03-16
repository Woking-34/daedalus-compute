// uniforms used by the fragment shader
uniform sampler3D volumeTex;
uniform sampler1D tffTex;

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
		vec4 entryPoint = tNear * viewDirW + eyePosW;
		vec4 exitPoint = tFar * viewDirW + eyePosW;
		
		vec4 voxelCoord = entryPoint;
		
		for( int i = 0; i < 256; ++i )
		{
			float intensity =  texture3D(volumeTex, voxelCoord.xyz).x;
			vec4 tffSample =  texture1D(tffTex, intensity);

			if(tffSample.a > 0.0)
			{
    			tffSample.a = 1.0 - pow(1.0 - tffSample.a, length(viewDirW*0.01)*200.0);
    			dst.rgb += (1.0 - dst.a) * tffSample.rgb * tffSample.a;
    			dst.a += (1.0 - dst.a) * tffSample.a;
    		}
						
			voxelCoord += viewDirW * 0.01;
			
			if(voxelCoord.x > 1.0 || voxelCoord.y > 1.0 || voxelCoord.z > 1.0)
			{
				dst.rgb = dst.rgb*dst.a + (1.0 - dst.a)*bgColor.rgb;
				break;
			}
				
			if(voxelCoord.x < 0.0 || voxelCoord.y < 0.0 || voxelCoord.z < 0.0)
			{
				dst.rgb = dst.rgb*dst.a + (1.0 - dst.a)*bgColor.rgb;
				break;
			}
		}
		
		gl_FragColor = dst;
		return;
	}
	
	discard;
}