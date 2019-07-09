#ifndef QUADRILATERAL_H
#define QUADRILATERAL_H

#include "math/mathbase.h"
#include "math/core/vec.h"
#include "math/util/aabb.h"

namespace daedalus {

template<typename T> struct Quadrilateral
{
	union
	{
		struct
		{
			T v[4];
		};

		struct
		{
			T a, b, c, d;
		};
	};

	T n;

	INLINE Quadrilateral() { }
	INLINE Quadrilateral(const T& a, const T& b, const T& c, const T& d) : a(a), b(b), c(c), d(d)
	{
		T edgeAB = b - a;
		T edgeAD = d - a;

		n = cross(edgeAB, edgeAD);
		normalize(n);
		n.w = -dot(n, a);
	}
 };

template<typename T> INLINE T center (const Quadrilateral<T>& quad) { return (quad.a  + quad.b + quad.c + quad.d)/T(4.0f); }

template<typename T> INLINE AABB<T> getAABB(const Quadrilateral<T>& quad)
{
	return AABB<T>( min( min(quad.a,quad.b), min(quad.c, quad.d) ), max( max(quad.a,quad.b), max(quad.c,quad.d) ) );
}

// acc data for parallelogram intersection

class Parallelogram
{
public:
	Vec4f plane, anchor;
	Vec4f v1, v2;

	INLINE Parallelogram() {}
	INLINE Parallelogram(const Vec4f& anchor, const Vec4f& offset1, const Vec4f& offset2) : anchor(anchor)
	{
		plane = normalize( cross( offset1, offset2 ) );
		plane.w = -dot( plane, anchor );

		v1 = offset1 / dot( offset1, offset1 );
		v2 = offset2 / dot( offset2, offset2 );
		v1.w = 1.0f;
		v2.w = 1.0f;
	}
};

} // namespace daedalus

#endif // QUADRILATERAL_H