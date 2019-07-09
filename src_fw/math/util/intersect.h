#ifndef INTERSECT_H
#define INTERSECT_H

#include "math/mathbase.h"
#include "math/core/vec.h"

#include "math/util/ray.h"
#include "math/util/aabb.h"
#include "math/util/obb.h"
#include "math/util/sphere.h"

#include "math/util/triangle.h"
#include "math/util/quadrilateral.h"

namespace daedalus {

class HitPoint4f
{
public:
	Vec4f p;
	Vec4f n;
};

bool intersectAABB( const Ray4f& ray, const AABB4f& aabb, float& tNear, float& tFar );
bool intersectSphere( const Ray4f& ray, const Sphere4f& sphere, float& tRay );

bool intersectTriEdgePlane( const Ray4f& ray, const TriEdgePlane& tri,  float& tRay, float& b1, float& b2);
bool intersectParallelogram( const Ray4f& ray, const Parallelogram& prlll, float& tRay, float& b1, float& b2);

} // namespace daedalus

#endif // INTERSECT_H