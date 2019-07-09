#ifndef AABB_H
#define AABB_H

#include "math/mathbase.h"
#include "math/core/vec.h"

namespace daedalus {

template<typename T> struct AABB
{
	T lower, upper;

	INLINE AABB() {}
	INLINE AABB(const AABB& box) { lower = box.lower; upper = box.upper; }

	INLINE AABB( const T& v                 ) : lower(v),   upper(v) {}
	INLINE AABB( const T& lower, const T& upper ) : lower(lower), upper(upper) {}

	INLINE AABB(FalseTy) : lower(pos_inf), upper(neg_inf) {}
	INLINE AABB(TrueTy)  : lower(neg_inf), upper(pos_inf) {}
};

template<typename T> INLINE T size(const AABB<T>& box) { return box.upper - box.lower; }
template<typename T> INLINE T center(const AABB<T>& box) { return T(0.5f)*(box.lower + box.upper); }

template<typename T> INLINE T area( const AABB<Vec2<T> >& b ) { const Vec2<T> d = size(b); return d.x*d.y; }
template<typename T> INLINE T area( const AABB<Vec4<T> >& b ) { const Vec4<T> d = size(b); return T(2.0f)*(d.x*d.y+d.x*d.z+d.y*d.z); }
template<typename T> INLINE T volume( const AABB<T>& b ) { return reduce_mul(size(b)); }

template<typename T> INLINE AABB<T> merge( const AABB<T>& a, const       T& b ) { return AABB<T>(min(a.lower, b    ), max(a.upper, b    )); }
template<typename T> INLINE AABB<T> merge( const       T& a, const AABB<T>& b ) { return AABB<T>(min(a    , b.lower), max(a    , b.upper)); }
template<typename T> INLINE AABB<T> merge( const AABB<T>& a, const AABB<T>& b ) { return AABB<T>(min(a.lower, b.lower), max(a.upper, b.upper)); }
template<typename T> INLINE AABB<T> merge( const AABB<T>& a, const AABB<T>& b, const AABB<T>& c, const AABB<T>& d) { return merge(merge(a,b),merge(c,d)); }
template<typename T> INLINE AABB<T>& operator+=( AABB<T>& a, const AABB<T>& b ) { return a = merge(a,b); }
template<typename T> INLINE AABB<T>& operator+=( AABB<T>& a, const       T& b ) { return a = merge(a,b); }

template<typename T> INLINE bool operator==( const AABB<T>& a, const AABB<T>& b ) { return a.lower == b.lower && a.upper == b.upper; }
template<typename T> INLINE bool operator!=( const AABB<T>& a, const AABB<T>& b ) { return a.lower != b.lower || a.upper != b.upper; }

template<typename T> INLINE AABB<T> operator *( const float& a, const AABB<T>& b ) { return AABB<T>(a*b.lower,a*b.upper); }

template<typename T> INLINE AABB<T> intersect( const AABB<T>& a, const AABB<T>& b ) { return AABB<T>(max(a.lower, b.lower), min(a.upper, b.upper)); }
template<typename T> INLINE AABB<T> intersect( const AABB<T>& a, const AABB<T>& b, const AABB<T>& c ) { return intersect(a,intersect(b,c)); }

/*! tests if bounding boxes (and points) are disjoint (empty intersection) */
template<typename T> INLINE bool disjoint( const AABB<T>& a, const AABB<T>& b )
{ const T d = min(a.upper, b.upper) - max(a.lower, b.lower); for ( size_t i = 0 ; i < T::N ; i++ ) if ( d[i] < typename T::Scalar(zero) ) return true; return false; }
template<typename T> INLINE bool disjoint( const AABB<T>& a, const  T& b )
{ const T d = min(a.upper, b)       - max(a.lower, b);       for ( size_t i = 0 ; i < T::N ; i++ ) if ( d[i] < typename T::Scalar(zero) ) return true; return false; }
template<typename T> INLINE bool disjoint( const  T& a, const AABB<T>& b )
{ const T d = min(a, b.upper)       - max(a, b.lower);       for ( size_t i = 0 ; i < T::N ; i++ ) if ( d[i] < typename T::Scalar(zero) ) return true; return false; }

/*! tests if bounding boxes (and points) are conjoint (non-empty intersection) */
template<typename T> INLINE bool conjoint( const AABB<T>& a, const AABB<T>& b )
{ const T d = min(a.upper, b.upper) - max(a.lower, b.lower); for ( size_t i = 0 ; i < T::N ; i++ ) if ( d[i] < typename T::Scalar(zero) ) return false; return true; }
template<typename T> INLINE bool conjoint( const AABB<T>& a, const  T& b )
{ const T d = min(a.upper, b)       - max(a.lower, b);       for ( size_t i = 0 ; i < T::N ; i++ ) if ( d[i] < typename T::Scalar(zero) ) return false; return true; }
template<typename T> INLINE bool conjoint( const  T& a, const AABB<T>& b )
{ const T d = min(a, b.upper)       - max(a, b.lower);       for ( size_t i = 0 ; i < T::N ; i++ ) if ( d[i] < typename T::Scalar(zero) ) return false; return true; }

/*! subset relation */
template<typename T> INLINE bool subset( const AABB<T>& a, const AABB<T>& b )
{ 
	for ( size_t i = 0 ; i < T::N ; i++ ) if ( a.lower[i]*1.00001f < b.lower[i] ) return false; 
	for ( size_t i = 0 ; i < T::N ; i++ ) if ( a.upper[i] > b.upper[i]*1.00001f ) return false; 
	return true; 
}

/*! default template instantiations */
typedef AABB< Vec2f > AABB2f;
typedef AABB< Vec4f > AABB4f;

typedef AABB< Vec2d > AABB2d;
typedef AABB< Vec4d > AABB4d;

} // namespace daedalus

#endif // AABB_H