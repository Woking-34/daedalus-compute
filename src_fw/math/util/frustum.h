#ifndef FRUSTUM_H
#define FRUSTUM_H

#include "math/mathbase.h"
#include "math/base/vec.h"
#include "math/base/mat.h"

template<typename T> struct Frustum
{
	T left, right, bottom, top, near, far;

	Mat44<T> projectionMatrix;			// glOrtho || glFrustum() || gluPerspective()
										// orthographic viewing frustum || perspective viewing frustum

	Vec4<T> cornerPoints[8];
	Vec4<T> boundingPlanes[6];

	INLINE Frustum() {}
};

template<typename T> INLINE Frustum<T> createProjectionPerspectiveFrustum(const T _left, const T _right, const T _bottom, const T _top, const T _near, const T _far)
{
	Frustum<T> f;

	f.projectionMatrix = createProjectionPerspectiveMatrix(_left, _right, _bottom, _top, _near, _far);
	/*
	Mat44<T> invmat = getMatrixtInverse(f.projectionMatrix);
	for(int i = 0; i < 8; ++i)
	{
		cornerPoints[i] = unitcubeVertices[i] * invmat;
	}

	for(int i = 0; i < 4; ++i)
	{
		boundingPlanes[i] = createPlane( cornerPoints[ cubeEdgeConnections[i].x ], cornerPoints[ cubeEdgeConnections[i].y ], cornerPoints[ cubeEdgeConnections[i].x+4 ] );
	}
	boundingPlanes[4] = createPlane( cornerPoints[ 0 ], cornerPoints[ 1 ], cornerPoints[ 3 ] );
	boundingPlanes[5] = createPlane( cornerPoints[ 6 ], cornerPoints[ 5 ], cornerPoints[ 7 ] );
	*/
	return f;
}

template<typename T> INLINE Frustum<T> createProjectionPerspectiveFrustum(const T _fovY, const T _aspectRatio, const T _near, const T _far)
{
	Frustum<T> f;

	f.projectionMatrix = createProjectionPerspectiveMatrix(_fovY, _aspectRatio, _near, _far);
	/*
	Mat44<T> invmat = getMatrixtInverse(f.projectionMatrix);
	for(int i = 0; i < 8; ++i)
	{
		cornerPoints[i] = unitcubeVertices[i] * invmat;
	}

	for(int i = 0; i < 4; ++i)
	{
		boundingPlanes[i] = createPlane( cornerPoints[ cubeEdgeConnections[i].x ], cornerPoints[ cubeEdgeConnections[i].y ], cornerPoints[ cubeEdgeConnections[i].x+4 ] );
	}
	boundingPlanes[4] = createPlane( cornerPoints[ 0 ], cornerPoints[ 1 ], cornerPoints[ 3 ] );
	boundingPlanes[5] = createPlane( cornerPoints[ 5 ], cornerPoints[ 4 ], cornerPoints[ 6 ] );
	*/
	return f;
}

#endif
