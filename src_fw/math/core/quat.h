#ifndef QUATERNION_H
#define QUATERNION_H

#include "math/mathbase.h"

namespace daedalus {

template<typename T> struct Quat
{
	T r, i, j, k;

	INLINE Quat() {}
    INLINE Quat(const Quat& q) { r = q.r; i = q.i; j = q.j; k = q.k; }

    INLINE Quat( const T& r, const T& i, const T& j , const T& k ) : r(r), i(i), j(j), k(k) {}
    INLINE Quat( const T& r, const Vec4<T>& v ) : r(r), i(v.x), j(v.y), k(v.z) {}

    //INLINE Quat( const Vec4<T>& vx, const Vec4<T>& vy, const Vec4<T>& vz );
    //INLINE Quat( const T& yaw, const T& pitch, const T& roll );
};

} // namespace daedalus

#endif // QUATERNION_H
