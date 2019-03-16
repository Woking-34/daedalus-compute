#ifndef MATHBASE_H
#define MATHBASE_H

#include "system/constants.h"

template<typename T> INLINE T sqr(const T& x) { return x*x; }
template<typename T> INLINE T rcp(const T& x) { return T(one)/x; }
template<typename T> INLINE T rsqrt(const T& x) { return T(one)/sqrt(x); }

template<typename T> INLINE T sign(const T& x) { return x<T(zero) ? -T(one) : T(one); }
template<typename T> INLINE bool samesign(const T& x, const T& y) { return x*y >= T(zero); } 

template<typename T> INLINE T min(const T& a, const T& b)                                     { return a<b? a:b; }
template<typename T> INLINE T min(const T& a, const T& b, const T& c, const T& d)             { return min(min(a,b),min(c,d)); }

template<typename T> INLINE T max(const T& a, const T& b)                                     { return a<b? b:a; }
template<typename T> INLINE T max(const T& a, const T& b, const T& c, const T& d)             { return max(max(a,b),max(c,d)); }

template<typename T> INLINE T clamp(const T& x, const T& lower = T(zero), const T& upper = T(one)) { return max(lower, min(x,upper)); }
template<typename T> INLINE uint32 quantize(const T& x, const uint32 n) { return clamp( int32( x * T(n) ), int32(0), int32(n-1));  }

template<typename T> INLINE T lerp(const T& t, const T& v1, const T& v2) { return (T(one) - t) * v1 + t * v2; }

template<typename T> INLINE T deg2rad ( const T& x )  { return x * T(1.74532925199432957692e-2f); }
template<typename T> INLINE T rad2deg ( const T& x )  { return x * T(5.72957795130823208768e1f); }
template<typename T> INLINE T sin2cos ( const T& x )  { return sqrt(max(T(zero),T(one)-x*x)); }
template<typename T> INLINE T cos2sin ( const T& x )  { return sin2cos(x); }

/*! random functions */
//template<typename T> INLINE T random() { return T(0); }
//template<> INLINE int32  random() { return int32(rand()); }
//template<> INLINE uint32 random() { return uint32(rand()); }
//template<> INLINE float  random() { return random<uint32>()/float(RAND_MAX); }
//template<> INLINE double random() { return random<uint32>()/double(RAND_MAX); }

/*! selects */
template<typename T> INLINE bool select ( const bool s, const T& t, const T& f )  { return s ? t : f; }

/* bit operatnds */
// clz
// popcnt

inline uint32 morton_code(uint32 x, uint32 y, uint32 z)
{
    x = (x | (x << 16)) & 0x030000FF; 
    x = (x | (x <<  8)) & 0x0300F00F; 
    x = (x | (x <<  4)) & 0x030C30C3; 
    x = (x | (x <<  2)) & 0x09249249; 

    y = (y | (y << 16)) & 0x030000FF; 
    y = (y | (y <<  8)) & 0x0300F00F; 
    y = (y | (y <<  4)) & 0x030C30C3; 
    y = (y | (y <<  2)) & 0x09249249; 

    z = (z | (z << 16)) & 0x030000FF; 
    z = (z | (z <<  8)) & 0x0300F00F; 
    z = (z | (z <<  4)) & 0x030C30C3; 
    z = (z | (z <<  2)) & 0x09249249; 

    return x | (y << 1) | (z << 2);
}

inline void fabsArray(float* data, int numelements)
{
	for (int i = 0; i < numelements; ++i)
	{
		data[i] = fabs(data[i]);
	}
}

inline void minmaxArray(float* data, int numelements, float& minValue, float& maxValue)
{
	minValue = FLT_MAX;
	maxValue = FLT_MIN;

	for(int i = 0; i < numelements; ++i)
	{
		minValue = std::min(minValue, data[i]);
		maxValue = std::max(maxValue, data[i]);
	}
}

inline void normalizeArray(float* data, int numelements)
{
	float minValue, maxValue;
	minmaxArray(data, numelements, minValue, maxValue);

	for (int i = 0; i < numelements; ++i)
	{
		data[i] = (data[i] - minValue) / (maxValue - minValue);
	}
}

#endif 
