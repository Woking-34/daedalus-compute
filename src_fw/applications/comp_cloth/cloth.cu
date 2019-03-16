#include <cuda.h>
#include <cuda_runtime_api.h>

#include <math.h>
#include <float.h>

#include <vector_types.h>
#include <vector_functions.h>

#include "cutil_math.h"

#define KSSTRUCT 50.75f
#define KDSTRUCT -0.25f
#define KSSHEAR 50.75f
#define KDSHEAR -0.25f
#define KSBEND 50.95f
#define KDBEND -0.25f

__constant__ float KsStruct = KSSTRUCT;
__constant__ float KdStruct = KDSTRUCT;
__constant__ float KsShear = KSSHEAR;
__constant__ float KdShear = KDSHEAR;
__constant__ float KsBend = KSBEND;
__constant__ float KdBend = KDBEND;

__constant__ float springStiffnes[12] =
{
	KSSTRUCT,
	KSSTRUCT,
	KSSTRUCT,
	KSSTRUCT,
	KSSHEAR,
	KSSHEAR,
	KSSHEAR,
	KSSHEAR,
	KSBEND,
	KSBEND,
	KSBEND,
	KSBEND
};

__constant__ float springDamping[12] =
{
	KDSTRUCT,
	KDSTRUCT,
	KDSTRUCT,
	KDSTRUCT,
	KDSHEAR,
	KDSHEAR,
	KDSHEAR,
	KDSHEAR,
	KDBEND,
	KDBEND,
	KDBEND,
	KDBEND
};

__constant__ int springCoord[2 * 12] =
{
	1, 0,
	0, -1,
	-1, 0,
	0, 1,
	1, -1,
	-1, -1,
	-1, 1,
	1, 1,
	2, 0,
	0, -2,
	-2, 0,
	0, 2
};

//structural springs (adjacent neighbors)
//        o
//        |
//     o--m--o
//        |
//        o

//shear springs (diagonal neighbors)
//     o  o  o
//      \   /
//     o  m  o
//      /   \
//     o  o  o

//bend spring (adjacent neighbors 1 node away)
//
//o   o   o   o   o
//        | 
//o   o   |   o   o
//        |   
//o-------m-------o
//        |  
//o   o   |   o   o
//        |
//o   o   o   o   o

__forceinline__ __device__ void getSpringCoord(int k, int* x, int* y)
{
	*x = springCoord[k * 2 + 0];
	*y = springCoord[k * 2 + 1];
}

__forceinline__ __device__ void getSpringCoeff(int k, float* ks, float* kd)
{
	*ks = springStiffnes[k];
	*kd = springDamping[k];
}

extern "C"
__global__
void verlet(const float4* __restrict__ g_pos_in, const float4* __restrict__ g_pos_old_in, float4* __restrict__ g_pos_out, float4* __restrict__ g_pos_old_out, float4* __restrict__ g_normals_out,
			const float* __restrict__ g_mass_in, float damp, float dt, float stepX, float stepY)
{
	// workitem/worksize info
	int idX = blockIdx.x * blockDim.x + threadIdx.x;
	int idY = blockIdx.y * blockDim.y + threadIdx.y;

	int sizeX = gridDim.x*blockDim.x;
	int sizeY = gridDim.y*blockDim.y;

	int index = (idY * sizeX) + idX;

	float mass = g_mass_in[index];

	float4 pos = g_pos_in[index];
	float4 pos_old = g_pos_old_in[index];
	float4 vel = (pos - pos_old) / dt;

	const float4 gravity = make_float4(0.0f, -0.00981f, 0.0f, 0.0f);
	float4 force = gravity*mass + vel*damp;

	float ks, kd;
	int x, y;

	for (int k = 0; k < 12; ++k)
	{
		getSpringCoord(k, &x, &y);
		getSpringCoeff(k, &ks, &kd);

		if (((idX + x) < (int)0) || ((idX + x) > (sizeX - 1)))
			continue;

		if (((idY + y) < (int)0) || ((idY + y) > (sizeY - 1)))
			continue;

		int index_neigh = (idY + y) * sizeX + (idX + x);

		float rest_length = length(make_float2(x * stepX, y * stepY));

		float4 pos2 = g_pos_in[index_neigh];
		float4 pos2_old = g_pos_old_in[index_neigh];
		float4 vel2 = (pos2 - pos2_old) / dt;

		float4 deltaP = pos - pos2;
		float4 deltaV = vel - vel2;
		float dist = length(deltaP);

		float leftTerm = -ks * (dist - rest_length);
		float rightTerm = kd * (dot(deltaV, deltaP) / dist);
		float4 springForce = (leftTerm + rightTerm)*normalize(deltaP);
		force += springForce;
	}

	float4 normal;

	{
		int index_neigh_left = (idY)* sizeX + max((idX - 1), 0);
		int index_neigh_right = (idY)* sizeX + min((idX + 1), sizeX - 1);
		int index_neigh_bottom = max((idY - 1), 0) * sizeX + (idX);
		int index_neigh_top = min((idY + 1), sizeY - 1) * sizeX + (idX);

		float4 left = g_pos_in[index_neigh_left];
		float4 right = g_pos_in[index_neigh_right];
		float4 bottom = g_pos_in[index_neigh_bottom];
		float4 top = g_pos_in[index_neigh_top];

		float4 tangentX = right - left;

		if (dot(tangentX, tangentX) < 1e-10f)
			tangentX = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
		else
			tangentX = normalize(tangentX);

		float4 tangentZ = bottom - top;

		if (dot(tangentZ, tangentZ) < 1e-10f)
			tangentZ = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
		else
			tangentZ = normalize(tangentZ);

		normal = make_float4(cross(make_float3(tangentX), make_float3(tangentZ)), 0.0f);
	}

	float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (mass != 0.0f)
		acc = force / mass;

	// verlet
	float4 tmp = pos;
	pos = pos * 2.0f - pos_old + acc * dt * dt;
	pos_old = tmp;

	float cf = 0.75f;
	float4 d = pos - pos_old;
	float4 rt = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	// y-up world plane
	{
		if (pos.y < 0.0f)
		{
			// collision
			float4 coll_dir = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
			pos.y = 0.0f;

			float4 dt = d - coll_dir * dot(d, coll_dir);
			rt += -cf*dt;
		}
	}

	// sphere
	{
		float4 center = make_float4(0.0f, 2.0f, 0.0f, 1.0f);
		float radius = 1.75f;

		if (length(pos - center) < radius)
		{
			// collision
			float4 coll_dir = normalize(pos - center);
			pos = center + coll_dir * radius;

			float4 dt = d - coll_dir * dot(d, coll_dir);
			rt += -cf*dt;
		}
	}

	g_pos_out[index] = pos + rt;
	g_pos_old_out[index] = pos_old;

	g_normals_out[index] = normalize(normal);
}

extern "C"
void cuK_cloth
(
	void* g_pos_in, void* g_pos_old_in, void* g_pos_out, void* g_pos_old_out, void* g_normals_out,
	void* mass, float damp, float dt, float stepX, float stepY,
	int gSizeX, int gSizeY, int lSizeX, int lSizeY
)
{
	dim3 block(lSizeX, lSizeY, 1);
	dim3 grid(gSizeX / block.x, gSizeY / block.y, 1);

	verlet <<< grid, block >>>((float4*)g_pos_in, (float4*)g_pos_old_in, (float4*)g_pos_out, (float4*)g_pos_old_out, (float4*)g_normals_out, (float*)mass, damp, dt, stepX, stepY);
}