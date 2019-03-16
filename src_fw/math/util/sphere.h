#ifndef SPHERE_H
#define SPHERE_H

#include "math/mathbase.h"
#include "math/core/vec.h"

template<typename T> struct Sphere
{
	T org;
	typename T::Scalar rad;

	INLINE Sphere() {}
	INLINE Sphere(const Sphere& r) { org = r.org; rad = r.rad; }

	INLINE Sphere( const T& org, const typename T::Scalar& rad ) : org(org), rad(rad) {}
};

typedef Sphere<Vec4f> Sphere4f;

#endif
