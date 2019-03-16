#ifndef RAY_H
#define RAY_H

#include "math/mathbase.h"
#include "math/core/vec.h"

template<typename T> struct Ray
{
	T org, dir;

	INLINE Ray() {}
	INLINE Ray(const Ray& r) { org = r.org; dir = r.dir; }

	INLINE Ray( const T& org, const T& dir ) : org(org), dir(dir) {}
};

typedef Ray<Vec4f> Ray4f;

#endif
