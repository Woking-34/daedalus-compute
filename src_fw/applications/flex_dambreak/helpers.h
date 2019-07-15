// This code contains NVIDIA Confidential Information and is disclosed to you
// under a form of NVIDIA software license agreement provided separately to you.
//
// Notice
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software and related documentation and
// any modifications thereto. Any use, reproduction, disclosure, or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA Corporation is strictly prohibited.
//
// ALL NVIDIA DESIGN SPECIFICATIONS, CODE ARE PROVIDED "AS IS.". NVIDIA MAKES
// NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.
//
// Information and code furnished is believed to be accurate and reliable.
// However, NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2013-2017 NVIDIA Corporation. All rights reserved.

#pragma once

#include <stdarg.h>

// disable some warnings
#if _WIN32
#pragma warning(disable: 4267)  // conversion from 'size_t' to 'int', possible loss of data
#endif

float SampleSDF(const float* sdf, int dim, int x, int y, int z)
{
	assert(x < dim && x >= 0);
	assert(y < dim && y >= 0);
	assert(z < dim && z >= 0);

	return sdf[z*dim*dim + y*dim + x];
}

// return normal of signed distance field
Vec3 SampleSDFGrad(const float* sdf, int dim, int x, int y, int z)
{
	int x0 = std::max(x-1, 0);
	int x1 = std::min(x+1, dim-1);

	int y0 = std::max(y-1, 0);
	int y1 = std::min(y+1, dim-1);

	int z0 = std::max(z-1, 0);
	int z1 = std::min(z+1, dim-1);

	float dx = (SampleSDF(sdf, dim, x1, y, z) - SampleSDF(sdf, dim, x0, y, z))*(dim*0.5f);
	float dy = (SampleSDF(sdf, dim, x, y1, z) - SampleSDF(sdf, dim, x, y0, z))*(dim*0.5f);
	float dz = (SampleSDF(sdf, dim, x, y, z1) - SampleSDF(sdf, dim, x, y, z0))*(dim*0.5f);

	return Vec3(dx, dy, dz);
}

void GetParticleBounds(Vec3& lower, Vec3& upper)
{
	lower = Vec3(FLT_MAX);
	upper = Vec3(-FLT_MAX);

	for (int i=0; i < g_buffers->positions.size(); ++i)
	{
		lower = Min(Vec3(g_buffers->positions[i]), lower);
		upper = Max(Vec3(g_buffers->positions[i]), upper);
	}
}


void CreateParticleGrid(Vec3 lower, int dimx, int dimy, int dimz, float radius, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, float jitter=0.005f)
{
	if (rigid && g_buffers->rigidIndices.empty())
		g_buffers->rigidOffsets.push_back(0);

	for (int x = 0; x < dimx; ++x)
	{
		for (int y = 0; y < dimy; ++y)
		{
			for (int z=0; z < dimz; ++z)
			{
				if (rigid)
					g_buffers->rigidIndices.push_back(int(g_buffers->positions.size()));

				Vec3 position = lower + Vec3(float(x), float(y), float(z))*radius + RandomUnitVector()*jitter;

				g_buffers->positions.push_back(Vec4(position.x, position.y, position.z, invMass));
				g_buffers->velocities.push_back(velocity);
				g_buffers->phases.push_back(phase);
			}
		}
	}

	if (rigid)
	{
		g_buffers->rigidCoefficients.push_back(rigidStiffness);
		g_buffers->rigidOffsets.push_back(int(g_buffers->rigidIndices.size()));
	}
}

void ClearShapes()
{
	g_buffers->shapeGeometry.resize(0);
	g_buffers->shapePositions.resize(0);
	g_buffers->shapeRotations.resize(0);
	g_buffers->shapePrevPositions.resize(0);
	g_buffers->shapePrevRotations.resize(0);
	g_buffers->shapeFlags.resize(0);
}

void UpdateShapes()
{	
	// mark shapes as dirty so they are sent to flex during the next update
	g_shapesChanged = true;
}

// calculates the union bounds of all the collision shapes in the scene
void GetShapeBounds(Vec3& totalLower, Vec3& totalUpper)
{
	Bounds totalBounds;

	for (int i=0; i < g_buffers->shapeFlags.size(); ++i)
	{
		NvFlexCollisionGeometry geo = g_buffers->shapeGeometry[i];

		int type = g_buffers->shapeFlags[i]&eNvFlexShapeFlagTypeMask;

		Vec3 localLower;
		Vec3 localUpper;

		switch(type)
		{
			case eNvFlexShapeBox:
			{
				localLower = -Vec3(geo.box.halfExtents);
				localUpper = Vec3(geo.box.halfExtents);
				break;
			}
			case eNvFlexShapeSphere:
			{
				localLower = -geo.sphere.radius;
				localUpper = geo.sphere.radius;
				break;
			}
			case eNvFlexShapeCapsule:
			{
				localLower = -Vec3(geo.capsule.halfHeight, 0.0f, 0.0f) - Vec3(geo.capsule.radius);
				localUpper = Vec3(geo.capsule.halfHeight, 0.0f, 0.0f) + Vec3(geo.capsule.radius);
				break;
			}
			case eNvFlexShapeConvexMesh:
			{
				NvFlexGetConvexMeshBounds(g_flexLib, geo.convexMesh.mesh, localLower, localUpper);

				// apply instance scaling
				localLower *= geo.convexMesh.scale;
				localUpper *= geo.convexMesh.scale;
				break;
			}
			case eNvFlexShapeTriangleMesh:
			{
				NvFlexGetTriangleMeshBounds(g_flexLib, geo.triMesh.mesh, localLower, localUpper);
				
				// apply instance scaling
				localLower *= Vec3(geo.triMesh.scale);
				localUpper *= Vec3(geo.triMesh.scale);
				break;
			}
			case eNvFlexShapeSDF:
			{
				localLower = 0.0f;
				localUpper = geo.sdf.scale;
				break;
			}
		};

		// transform local bounds to world space
		Vec3 worldLower, worldUpper;
		TransformBounds(localLower, localUpper, Vec3(g_buffers->shapePositions[i]), g_buffers->shapeRotations[i], 1.0f, worldLower, worldUpper);

		totalBounds = Union(totalBounds, Bounds(worldLower, worldUpper));
	}

	totalLower = totalBounds.lower;
	totalUpper = totalBounds.upper;
}