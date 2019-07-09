#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "math/mathbase.h"
#include "math/core/vec.h"
#include "math/util/aabb.h"

namespace daedalus {

template<typename T> struct Triangle
{
	union
	{
		struct
		{
			T v[3];
		};

		struct
		{
			T a, b, c;
		};
	};

	T n;

	INLINE Triangle() { }
	INLINE Triangle(const T& a, const T& b, const T& c) : a(a), b(b), c(c)
	{
		T edgeAB = b - a;
		T edgeAC = c - a;

		n = cross(edgeAB, edgeAC);
		n = normalize(n);
		n.w = -dot(n, a);
	}
 };

template<typename T> INLINE T center (const Triangle<T>& tri) { return (tri.a  + tri.b + tri.c)/T(3.0f); }

template<typename T> INLINE AABB<T> getAABB(const Triangle<T>& tri)
{
	return AABB<T>( min(min(tri.a,tri.b),tri.c), max(max(tri.a,tri.b),tri.c) );
}

// acc data for ray intersection

class TriEdgePlane
{
public:
	Vec4f plane, ep20, ep01;

	INLINE TriEdgePlane() {}
	INLINE TriEdgePlane(const Vec4f& vertexA, const Vec4f& vertexB, const Vec4f& vertexC)
	{
		plane = normalize( cross(vertexB - vertexA, vertexC - vertexA) );
		plane.w = -dot(plane, vertexA);
	
		ep20 = normalize( cross(plane, vertexA-vertexC) );
		ep20.w = -dot(ep20, vertexC);
	
		ep01 = normalize( cross(plane, vertexB-vertexA) );
		ep01.w = -dot(ep01, vertexA);
	
		ep20 *= 1.0f / dot(vertexB, ep20);
		ep01 *= 1.0f / dot(vertexC, ep01);
	}
};

} // namespace daedalus

#endif // TRIANGLE_H