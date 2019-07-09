#include "intersect.h"

namespace daedalus {

bool intersectAABB( const Ray4f& ray, const AABB4f& aabb, float& tNear, float& tFar)
{	
	const Vec4f dirInv= rcp(ray.dir);

	const Vec4f tnear4 = dirInv * (aabb.lower - ray.org);
    const Vec4f tfar4 = dirInv * (aabb.upper - ray.org);

    const Vec4f t0 = min(tnear4, tfar4);
    const Vec4f t1 = max(tnear4, tfar4);

    tNear = max(max(t0.x, t0.y), t0.z);
    tFar = min(min(t1.x, t1.y), t1.z);

    return (tFar >= tNear) && (tFar > 0.0f);
}

bool intersectSphere( const Ray4f& ray, const Sphere4f& sphere, float& tRay )
{
	const Vec4f SphToRay = ray.org - sphere.org;

	const float a = dot(ray.dir, ray.dir);
	const float b = 2.0f * dot(ray.dir, SphToRay);
	const float c = dot(SphToRay, SphToRay) - sphere.rad*sphere.rad;

	float disc = (b*b - 4*a*c);

	if( disc < 0.0f )
	{
		return 0;
	}

	disc = std::sqrt(disc);

	tRay = (-b - disc) / (2.0f*a);
	if(tRay > 0.0f)
	{
		return 1;
	}

	tRay = (-b + disc) / (2.0f*a);
	if(tRay > 0.0f)
	{
		return 1;
	}

	return 0;
}

bool intersectTriEdgePlane( const Ray4f& ray, const TriEdgePlane& tri, float& tRay, float& b1, float& b2)
{
	const float denom = dot(tri.plane, ray.dir);

	tRay = (-1.0f * dot(tri.plane, ray.org)) / denom;

	const Vec4f intPt = ray.org + tRay * ray.dir; 

	b1 = dot(intPt, tri.ep20);
	b2 = dot(intPt, tri.ep01);

	return b1 >= 0.0f && b2 >= 0.0f && b1 + b2 <= 1.0f && tRay > 0.0f;
}

bool intersectParallelogram( const Ray4f& ray, const Parallelogram& prlll, float& tRay, float& b1, float& b2)
{
	const float denom = dot(prlll.plane, ray.dir);

	tRay = (-1.0f * dot(prlll.plane, ray.org)) / denom;

	const Vec4f intPt = ray.org + tRay * ray.dir; 
	const Vec4f vi = intPt - prlll.anchor;

	b1 = dot(prlll.v1, vi);
	b2 = dot(prlll.v2, vi);

	return b1 >= 0.0f && b2 >= 0.0f && b1 <= 1.0f && b2 <= 1.0f && tRay > 0.0f;
}

} // namespace daedalus