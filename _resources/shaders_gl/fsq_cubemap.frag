// uniforms used by the fragment shader
uniform samplerCube cubeTex;
uniform vec4 eyePosW;

// input from vertex shader
varying vec4 v_PositionW;

bool intersectRaySphere(out float tRay, vec4 rayOrg, vec4 rayDir, vec4 sphPos, float sphRad)
{
	vec4 SphToRay = rayOrg - sphPos;
	
	float a = 1.0;//dot(rayDir, rayDir);
	float b = 2.0 * dot(rayDir, SphToRay);
	float c = dot(SphToRay, SphToRay) - sphRad*sphRad;
	
	float disc = (b*b - 4.0*a*c);
	
	if( disc < 0.0 )
	{
		return false;
	}
	
	disc = sqrt(disc);
	tRay = (-b - disc) / (2.0*a);
	
	return tRay > 0.0 ? true : false;
}

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
	
	//vec4 sphPos = vec4(0.0, 0.0, 0.0, 1.0);
	//float sphRad = 1.0;
	//float tRay;
	//
	//hasHit = intersectRaySphere(tRay, eyePosW, viewDirW, sphPos, sphRad);
	
	//vec4 aabbMin = vec4(-1.0,-1.0,-1.0, 1.0);
	//vec4 aabbMax = vec4( 1.0, 1.0, 1.0, 1.0);
	//float tNear, tFar;
	//
	//hasHit = intersectRayAABB(eyePosW, viewDirW, aabbMin, aabbMax, tNear, tFar);
	
	if(hasHit)
	{
		//vec4 hitPoint = eyePosW + viewDirW * tRay;
		//vec4 hitNormal = abs( normalize( hitPoint - sphPos ) );
		//
		//gl_FragColor = vec4(hitNormal.xyz, 1.0);
		
		//float maxDist = length( aabbMax - aabbMin );
		//float dist = tFar - tNear;
		//float color = dist / maxDist;
		//
		//gl_FragColor = vec4(color, color, color, 1.0);
	}
	else
	{
		gl_FragColor = textureCube(cubeTex, viewDirW.xyz);
	}
}