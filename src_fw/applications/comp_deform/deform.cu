#include <cuda.h>
#include <cuda_runtime_api.h>

#include <math.h>
#include <float.h>

#include <vector_types.h>
#include <vector_functions.h>

#include "cutil_math.h"

__constant__ float wave_a[2] = { 0.4f, 0.4f };
__constant__ float wave_k[2] = { 1.5f, 1.5f };
__constant__ float wave_w[2] = { 0.01f, 0.01f };
__constant__ float wave_p[2] = { 1.0f, 1.0f };
__constant__ float4 wave_dir = { 1.0f, 0.0f, -1.0f, 0.0f };

__forceinline__ __device__ float4 generatePosition(
	float idX, float idY, int currDeform, float currTime,
	float gridWidth, float gridWidthDt, float gridLength, float gridLengthDt)
{
	float4 ret;

	float x = idX * gridWidthDt - gridWidth * 0.5f;
	float y = 0.0f;
	float z = idY * gridLengthDt - gridLength * 0.5f;

	// Directional
	if (currDeform == 0)
	{
		float dist = dot(wave_dir, make_float4(x, y, z, 0.0f));
		for (int i = 0; i < 2; i++)
		{
			y += wave_a[i] * sin(wave_k[i] * dist - currTime*wave_w[i] + wave_p[i]);
		}
	}

	// Circular
	if (currDeform == 1)
	{
		float dist = sqrt(x*x + z*z);
		for (int i = 0; i < 2; i++)
		{
			y += wave_a[i] * sin(wave_k[i] * dist - currTime*wave_w[i] + wave_p[i]);
		}
	}

	// SinCos
	if (currDeform == 2)
	{
		y = cos(z + -0.5f * currTime*wave_w[0]) + 0.5f * sin(x + 2.0f * z - 0.5f * currTime*wave_w[1]);
	}

	ret.x = x;
	ret.y = y;
	ret.z = z;
	ret.w = 1.0f;

	return ret;
}

extern "C"
__global__
void deform
(
	float4* __restrict__ gridVerts, const int currDeform, const float currTime,
	const float gridWidth, const float gridWidthDt, const float gridLength, const float gridLengthDt
)
{
	// workitem/worksize info
	int idX = blockIdx.x * blockDim.x + threadIdx.x;
	int idY = blockIdx.y * blockDim.y + threadIdx.y;

	int sizeX = gridDim.x*blockDim.x;
	//int sizeY = gridDim.y*blockDim.y;

	float idXF = (float)(idX);
	float idYF = (float)(idY);

	float4 pos = generatePosition(idXF, idYF, currDeform, currTime, gridWidth, gridWidthDt, gridLength, gridLengthDt);

	float4 normal;

	{
		float4 left = generatePosition(idXF - gridWidth, idYF, currDeform, currTime, gridWidth, gridWidthDt, gridLength, gridLengthDt);
		float4 right = generatePosition(idXF + gridWidth, idYF, currDeform, currTime, gridWidth, gridWidthDt, gridLength, gridLengthDt);
		float4 bottom = generatePosition(idXF, idYF - gridLengthDt, currDeform, currTime, gridWidth, gridWidthDt, gridLength, gridLengthDt);
		float4 top = generatePosition(idXF, idYF + gridLengthDt, currDeform, currTime, gridWidth, gridWidthDt, gridLength, gridLengthDt);

		float4 tangentX = right - left;

		if (dot(tangentX, tangentX) < 1e-10f)
			tangentX = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
		else
			tangentX = normalize(tangentX);

		float4 tangentY = bottom - top;

		if (dot(tangentY, tangentY) < 1e-10f)
			tangentY = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
		else
			tangentY = normalize(tangentY);


		normal = cross(tangentX, tangentY);
		//normal = abs(normal);
	}

	gridVerts[2 * (idY * sizeX + idX) + 0] = pos;
	gridVerts[2 * (idY * sizeX + idX) + 1] = normal;
}

extern "C"
void cuK_deform
(
	void* g_pos, const int currDeform, const float currTime,
	const float gridWidth, const float gridWidthDt, const float gridLength, const float gridLengthDt,
	int gSizeX, int gSizeY, int lSizeX, int lSizeY
)
{
	dim3 block(lSizeX, lSizeY, 1);
	dim3 grid(gSizeX / block.x, gSizeY / block.y, 1);

	deform <<< grid, block >>>((float4*)g_pos, currDeform, currTime, gridWidth, gridWidthDt, gridLength, gridLengthDt);
}