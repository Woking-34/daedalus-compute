#include <cuda.h>
#include <cuda_runtime_api.h>

#include <math.h>
#include <float.h>

#include <vector_types.h>
#include <vector_functions.h>

#include "cutil_math.h"

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

extern "C"
__global__
void resetHeadList(int* __restrict__ hList)
{
	// workitem/worksize info
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	hList[index] = -1;
}

extern "C"
__global__
void createList(const float4* __restrict__ posBuffer, int* __restrict__ hList, int* __restrict__ pList,
				const float particleRadius, const int gridSize)
{
	// workitem/worksize info
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	float4 pos = posBuffer[index];
	
	int xId = clamp((int)(pos.x / (2.0f*particleRadius)), 0, gridSize-1);
	int yId = clamp((int)(pos.y / (2.0f*particleRadius)), 0, gridSize-1);
	int zId = clamp((int)(pos.z / (2.0f*particleRadius)), 0, gridSize-1);
	
	int gridId = zId * gridSize*gridSize + yId * gridSize + xId;
	
	int listId = atomicExch(hList + gridId, index);
	pList[index] = listId;
}

__forceinline__ __device__  void collideSpheres(float4 posA, float4 posB, float4 velA, float4 velB, float radiusA, float radiusB, float4* force)
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

extern "C"
__global__
void collideList(const int* __restrict__ hList, const int* __restrict__ pList,
				 const float4* __restrict__ posBuffer, const float4* __restrict__ vel0Buffer, float4* __restrict__ vel1Buffer,
				 const float particleRadius, const int gridSize)
{
	// workitem/worksize info
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	float4 pos = posBuffer[index];
	float4 vel = vel0Buffer[index];
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	
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

extern "C"
__global__
void integrate(float4* __restrict__ posBuffer, float4* __restrict__ vel0Buffer, const float4* __restrict__ vel1Buffer, const float particleRadius)
{
	// workitem/worksize info
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	float4 pos = posBuffer[index];
	float4 vel = vel1Buffer[index];
		
	float4 g = make_float4(0.0f, gravity, 0.0f, 0.0f);
	
	vel += g * deltaTime;
	
	vel *= globalDamping;
	
	pos += vel * deltaTime;
	
	if(pos.x < particleRadius)
	{
		pos.x = particleRadius;
		vel.x *= boundaryDamping;
	}
	
	if(pos.x > 1.0f - particleRadius)
	{
		pos.x = 1.0f - particleRadius;
		vel.x *= boundaryDamping;
	}
	
	if(pos.y < particleRadius)
	{
		pos.y = particleRadius;
		vel.y *= boundaryDamping;
	}
	
	if(pos.y > 1.0f - particleRadius)
	{
		pos.y = 1.0f - particleRadius;
		vel.y *= boundaryDamping;
	}
	
	if(pos.z < particleRadius)
	{
		pos.z = particleRadius;
		vel.z *= boundaryDamping;
	}
	
	if(pos.z > 1.0f - particleRadius)
	{
		pos.z = 1.0f - particleRadius;
		vel.z *= boundaryDamping;
	}
	
	posBuffer[index] = pos;
	vel0Buffer[index] = vel;
}

extern "C"
void cuK_reset
(
	void* hList,
	int gSizeX, int lSizeX
)
{
	dim3 block(lSizeX, 1, 1);
	dim3 grid(gSizeX / block.x, 1, 1);

	resetHeadList <<< grid, block >>>((int*)hList);
}

extern "C"
void cuK_create
(
	void* pos, void* hList, void* pList, float particleRadius, int gridSize,
	int gSizeX, int lSizeX
)
{
	dim3 block(lSizeX, 1, 1);
	dim3 grid(gSizeX / block.x, 1, 1);

	createList << < grid, block >> >((float4*)pos, (int*)hList, (int*)pList, particleRadius, gridSize);
}

extern "C"
void cuK_collide
(
	void* hList, void* pList, void* pos, void* vel0, void* vel1, float particleRadius, int gridSize,
	int gSizeX, int lSizeX
)
{
	dim3 block(lSizeX, 1, 1);
	dim3 grid(gSizeX / block.x, 1, 1);

	collideList << < grid, block >> >((int*)hList, (int*)pList, (float4*)pos, (float4*)vel0, (float4*)vel1, particleRadius, gridSize);
}

extern "C"
void cuK_integrate
(
	void* pos, void* vel0, void* vel1, float particleRadius,
	int gSizeX, int lSizeX
)
{
	dim3 block(lSizeX, 1, 1);
	dim3 grid(gSizeX / block.x, 1, 1);

	integrate << < grid, block >> >((float4*)pos, (float4*)vel0, (float4*)vel1, particleRadius);
}