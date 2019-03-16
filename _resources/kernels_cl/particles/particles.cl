#define spring				 0.5f
#define damping				 0.02f
#define shear				 0.1f
#define attraction			 0.0f
#define boundaryDamping		-0.5f
#define globalDamping		 1.0f
#define gravity				-0.03f
#define deltaTime			 0.01f

#ifndef M_PI_F
	#define M_PI_F 3.14159265358979323846f
#endif

__kernel
void resetHeadList(__global int* restrict hList)
{
	int index = get_global_id(0);
	hList[index] = -1;
}

__kernel
void createList(__global const float4* restrict posBuffer, __global int* restrict hList, __global int* restrict pList,
				const float particleRadius, const int gridSize)
{
	int index = get_global_id(0);
	
	float4 pos = posBuffer[index];
	
	int xId = clamp((int)(pos.x / (2.0f*particleRadius)), 0, gridSize-1);
	int yId = clamp((int)(pos.y / (2.0f*particleRadius)), 0, gridSize-1);
	int zId = clamp((int)(pos.z / (2.0f*particleRadius)), 0, gridSize-1);
	
	int gridId = zId * gridSize*gridSize + yId * gridSize + xId;
	
	int listId = atomic_xchg(hList + gridId, index);
	pList[index] = listId;
}

inline void collideSpheres(float4 posA, float4 posB, float4 velA, float4 velB, float radiusA, float radiusB, float4* force)
{
	float4 relPos = posB - posA;
	float dist = length(relPos);
	float collideDist = radiusA + radiusB;
	
	if(dist < collideDist)
	{
		float4 norm = normalize(relPos);
		float4 relVel = velB - velA;
		float4 tanVel = relVel - norm * dot(norm, relVel);
		
		*force = *force - norm * spring * (collideDist-dist);
		*force = *force + relVel * damping;
		*force = *force + tanVel * shear;
		*force = *force + relPos * attraction;
	}
}

__kernel
void collideList(__global const int* restrict hList, __global const int* restrict pList,
				 __global const float4* restrict posBuffer, __global const float4* restrict vel0Buffer, __global float4* restrict vel1Buffer,
				 const float particleRadius, const int gridSize)
{
	int index = get_global_id(0);
	
	float4 pos = posBuffer[index];
	float4 vel = vel0Buffer[index];
	float4 force = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	
	int xId = (int)(pos.x / (2.0f*particleRadius));
	int yId = (int)(pos.y / (2.0f*particleRadius));
	int zId = (int)(pos.z / (2.0f*particleRadius));
	
	int xIdMin = max(xId-1, 0);
	int yIdMin = max(yId-1, 0);
	int zIdMin = max(zId-1, 0);
	
	int xIdMax = min(xId+1, gridSize-1);
	int yIdMax = min(yId+1, gridSize-1);
	int zIdMax = min(zId+1, gridSize-1);
	
	for(int k = zIdMin; k <= zIdMax; ++k)
	{
		for(int j = yIdMin; j <= yIdMax; ++j)
		{
			for(int i = xIdMin; i <= xIdMax; ++i)
			{
				int gridId = k * gridSize*gridSize + j * gridSize + i;
				
				int listId = hList[gridId];
				
				while(listId != -1)
				{
					int listIdNew = pList[listId];
					
					if(index == listId)
					{
						listId = listIdNew;
						continue;
					}
					
					float4 pos2 = posBuffer[listId];
					float4 vel2 = vel0Buffer[listId];
					
					collideSpheres(pos, pos2, vel, vel2, particleRadius, particleRadius, &force);
					
					listId = listIdNew;
				}
			}
		}
	}
	
	vel1Buffer[index] = vel + force;
}

inline float4 quaternionConjugate(const float4 v)
{
	return (float4)(-v.s0, -v.s1, -v.s2, v.s3);
}

inline float4 quaternionMultiply(const float4 a, const float4 b)
{
	return (float4)
	(
		a.s3*b.s0 +a.s0*b.s3 +a.s1*b.s2 -a.s2*b.s1,
		a.s3*b.s1 -a.s0*b.s2 +a.s1*b.s3 +a.s2*b.s0,
		a.s3*b.s2 +a.s0*b.s1 -a.s1*b.s0 +a.s2*b.s3,
		a.s3*b.s3 -a.s0*b.s0 -a.s1*b.s1 -a.s2*b.s2
	);
}

inline float4 quaternionMultiplyVector(const float4 q, const float4 v)
{
	return (float4)
	(
		q.s3*v.s0 +q.s0*v.s3 +q.s1*v.s2 -q.s2*v.s1,
		q.s3*v.s1 -q.s0*v.s2 +q.s1*v.s3 +q.s2*v.s0,
		q.s3*v.s2 +q.s0*v.s1 -q.s1*v.s0 +q.s2*v.s3,
		0.0f
	);
}

inline float4 quaternionAngleRotator(const float angle, const float4 axis)
{
	float cosAngle;
	float sinAngle = sincos(angle*0.5f, &cosAngle);
	
	float4 r = axis;
	r.s0 *= sinAngle;
	r.s1 *= sinAngle;
	r.s2 *= sinAngle;
	r.s3  = cosAngle;
	
	return r;
}

inline float4 rotateVectorAxisAngle(const float4 v, const float angle, const float4 axis)
{
	float4 q = quaternionAngleRotator(angle, axis);
	return quaternionMultiplyVector(q, quaternionMultiply(v, quaternionConjugate(q)));
}

__kernel
void integrate(__global float4* restrict posBuffer, __global float4* restrict vel0Buffer, __global const float4* restrict vel1Buffer, const float particleRadius)
{
	int index = get_global_id(0);
	
	float4 pos = posBuffer[index];
	float4 vel = vel1Buffer[index];
		
	float4 g = (float4)(0.0f, gravity, 0.0f, 0.0f);
	
	vel += g * deltaTime;
	
	vel *= globalDamping;
	
	pos += vel * deltaTime;
	
	if(pos.s0 < particleRadius)
	{
		pos.s0 = particleRadius;
		vel.s0 *= boundaryDamping;
	}
	
	if(pos.s0 > 1.0f - particleRadius)
	{
		pos.s0 = 1.0f - particleRadius;
		vel.s0 *= boundaryDamping;
	}
	
	if(pos.s1 < particleRadius)
	{
		pos.s1 = particleRadius;
		vel.s1 *= boundaryDamping;
	}
	
	if(pos.s1 > 1.0f - particleRadius)
	{
		pos.s1 = 1.0f - particleRadius;
		vel.s1 *= boundaryDamping;
	}
	
	if(pos.s2 < particleRadius)
	{
		pos.s2 = particleRadius;
		vel.s2 *= boundaryDamping;
	}
	
	if(pos.s2 > 1.0f - particleRadius)
	{
		pos.s2 = 1.0f - particleRadius;
		vel.s2 *= boundaryDamping;
	}
	
	posBuffer[index] = pos;
	vel0Buffer[index] = vel;
}
