#ifndef VECTOR_H
#define VECTOR_H

#include "math/mathbase.h"

#define OP operator
#define DECL template<typename T> INLINE

#define V1 Vec1<T>
#define V2 Vec2<T>
#define V4 Vec4<T>
#define V1ARG const Vec1<T>&
#define V2ARG const Vec2<T>&
#define V4ARG const Vec4<T>&

//////////////////////////////////////////////////////////////////////////////
// 1D vector
//////////////////////////////////////////////////////////////////////////////

template<typename T> struct Vec1
{
	T x;

	typedef T Scalar;
	enum {N = 1};

	INLINE Vec1() {}
	INLINE Vec1(const Vec1& v) {x = v.x;}

	template<typename T1> INLINE Vec1 (const Vec1<T1>& a) : x(T(a.x)) {}
	template<typename T1> INLINE Vec1& operator= (const Vec1<T1>& v) { x = v.x; return *this; }

	INLINE explicit Vec1(const T& a) : x(a) {}
	INLINE          Vec1(const T* a) : x(a[0]) {}

	INLINE Vec1(ZeroTy) : x(zero) {}
	INLINE Vec1(PosInfTy) : x(pos_inf) {}
	INLINE Vec1(NegInfTy) : x(neg_inf) {}

	//INLINE const T& operator [](size_t elem) const { assert(elem < 1); return (&x)[elem]; }
	//INLINE       T& operator [](size_t elem)       { assert(elem < 1); return (&x)[elem]; }

	INLINE operator const T*() const { return (const T*)this; };
	INLINE operator       T*()       { return       (T*)this; };
};

//////////////////////////////////////////////////////////////////////////////
// 2D vector
//////////////////////////////////////////////////////////////////////////////

template<typename T> struct Vec2
{
	T x, y;

	typedef T Scalar;
	enum {N = 2};

	INLINE Vec2() {}
	INLINE Vec2(const Vec2& v) {x = v.x; y = v.y;}

	template<typename T1> INLINE Vec2 (const Vec2<T1>& a) : x(T(a.x)), y(T(a.y)) {}
	template<typename T1> INLINE Vec2& operator= (const Vec2<T1>& v) { x = v.x; y = v.y; return *this; }

	INLINE explicit Vec2(const T& a) : x(a), y(a) {}
	INLINE          Vec2(const T& x, const T& y) : x(x), y(y) {}
	INLINE          Vec2(const T* a, size_t stride = 1) : x(a[0]), y(a[stride]) {}

	INLINE Vec2(ZeroTy) : x(zero), y(zero) {}
	INLINE Vec2(PosInfTy) : x(pos_inf), y(pos_inf) {}
	INLINE Vec2(NegInfTy) : x(neg_inf), y(neg_inf) {}

	//INLINE const T& operator [](size_t elem) const { assert(elem < 2); return (&x)[elem]; }
	//INLINE       T& operator [](size_t elem)       { assert(elem < 2); return (&x)[elem]; }

	INLINE operator const T*() const { return (const T*)this; };
	INLINE operator       T*()       { return       (T*)this; };
};

DECL V2 OP+ (V2ARG a)  {return V2(+a.x, +a.y);}
DECL V2 OP- (V2ARG a)  {return V2(-a.x, -a.y);}

DECL V2 OP+ (V2ARG a, V2ARG b) {return V2(a.x + b.x, a.y + b.y);}
DECL V2 OP- (V2ARG a, V2ARG b) {return V2(a.x - b.x, a.y - b.y);}
DECL V2 OP* (V2ARG a, V2ARG b) {return V2(a.x * b.x, a.y * b.y);}
DECL V2 OP* (const T &a, V2ARG b)  {return V2(a * b.x, a * b.y);}
DECL V2 OP* (V2ARG a, const T &b)  {return V2(a.x * b, a.y * b);}
DECL V2 OP/ (V2ARG a, V2ARG b) {return V2(a.x / b.x, a.y / b.y);}
DECL V2 OP/ (V2ARG a, const T &b)  {return V2(a.x / b, a.y / b);}
DECL V2 OP/ (const T &a, V2ARG b)  {return V2(a / b.x, a / b.y);}

DECL V2& OP+= (V2& a, V2ARG b) {a.x += b.x; a.y += b.y; return a;}
DECL V2& OP-= (V2& a, V2ARG b) {a.x -= b.x; a.y -= b.y; return a;}
DECL V2& OP*= (V2& a, const T &b)  {a.x *= b; a.y *= b; return a;}
DECL V2& OP/= (V2& a, const T &b)  {a.x /= b; a.y /= b; return a;}

DECL bool OP== (V2ARG a, V2ARG b) {return a.x == b.x && a.y == b.y;}
DECL bool OP!= (V2ARG a, V2ARG b) {return a.x != b.x || a.y != b.y;}
DECL bool OP<  (V2ARG a, V2ARG b)
{
	if (a.x != b.x) return a.x < b.x;
	if (a.y != b.y) return a.y < b.y;
	return false;
}

DECL V2 abs (V2ARG a)  {return V2(fabs(a.x), fabs(a.y));}
DECL V2 rcp (V2ARG a)  {return V2(rcp(a.x), rcp(a.y));}
DECL V2 sqrt (V2ARG a) {return V2(sqrt(a.x), sqrt(a.y));}
DECL V2 rsqrt(V2ARG a) {return V2(rsqrt(a.x), rsqrt(a.y));}

DECL T reduce_add(V2ARG a) {return a.x + a.y;}
DECL T reduce_mul(V2ARG a) {return a.x * a.y;}
DECL T reduce_min(V2ARG a) {return min(a.x, a.y);}
DECL T reduce_max(V2ARG a) {return max(a.x, a.y);}

DECL V2 min(V2ARG a, V2ARG b)  {return V2(min(a.x, b.x), min(a.y, b.y));}
DECL V2 max(V2ARG a, V2ARG b)  {return V2(max(a.x, b.x), max(a.y, b.y));}

DECL T dot(V2ARG a, V2ARG b)   {return a.x*b.x + a.y*b.y;}
DECL T length(V2ARG a)      {return sqrt(dot(a,a));}
DECL T distance (V2ARG a, V2ARG b) {return length(a-b);}
DECL V2 normalize(V2ARG a)  {return a*rsqrt(dot(a,a));}

DECL V2 select( V2ARG s, V2ARG t, V2ARG f )
{
	return V2(select(s.x,t.x,f.x),
		select(s.y,t.y,f.y));
}
DECL V2 shuffle( V2ARG a, const Vec2<int>&	mask ) { return V2( a[mask.x], a[mask.y] ); }

DECL bool any( V2ARG a ) { return a.x || a.y; }
DECL bool all( V2ARG a ) { return a.x && a.y; }

//////////////////////////////////////////////////////////////////////////////
// 4D vector
//////////////////////////////////////////////////////////////////////////////

template<typename T> struct Vec4
{
	T x, y, z, w;

	typedef T Scalar;
	enum {N = 4};

	INLINE Vec4() {}
	INLINE Vec4(const Vec4& v) {x = v.x; y = v.y; z = v.z; w = v.w;}

	template<typename T1> INLINE Vec4 (const Vec4<T1>& a) : x(T(a.x)), y(T(a.y)), z(T(a.z)), w(T(a.w)) {}
	template<typename T1> INLINE Vec4& operator= (const Vec4<T1>& v) {x = v.x; y = v.y; z = v.z; w = v.w; return *this;}

	INLINE explicit Vec4(const T &a) : x(a), y(a), z(a), w(a) {}
	INLINE          Vec4(const T &x, const T &y, const T &z, const T &w) : x(x), y(y), z(z), w(w) {}
	INLINE          Vec4(const T* a, size_t stride = 1) : x(a[0]), y(a[stride]), z(a[2*stride]), w(a[3*stride]) {}

	INLINE Vec4(ZeroTy)   : x(zero), y(zero), z(zero), w(zero) {}
	INLINE Vec4(PosInfTy) : x(pos_inf), y(pos_inf), z(pos_inf), w(T(one)) {}
	INLINE Vec4(NegInfTy) : x(neg_inf), y(neg_inf), z(neg_inf), w(T(one)) {}

	//INLINE const T& operator [](size_t elem) const { assert(elem < 4); return (&x)[elem]; }
	//INLINE       T& operator [](size_t elem)       { assert(elem < 4); return (&x)[elem]; }

	INLINE operator const T*() const { return (const T*)this; };
	INLINE operator       T*()       { return       (T*)this; };
};

DECL V4 OP+ (V4ARG a) {return V4(+a.x, +a.y, +a.z, +a.w);}
DECL V4 OP- (V4ARG a) {return V4(-a.x, -a.y, -a.z, -a.w);}

DECL V4 OP+ (V4ARG a, V4ARG b) {return V4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);}
DECL V4 OP- (V4ARG a, V4ARG b) {return V4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);}
DECL V4 OP* (V4ARG a, V4ARG b) {return V4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);}
DECL V4 OP* (V4ARG a, const T &b) {return V4(a.x * b, a.y * b, a.z * b, a.w * b);}
DECL V4 OP* (const T &a, V4ARG b) {return V4(a * b.x, a * b.y, a * b.z, a * b.w);}
DECL V4 OP/ (V4ARG a, V4ARG b) {return V4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);}
DECL V4 OP/ (V4ARG a, const T &b) {return V4(a.x / b, a.y / b, a.z / b, a.w / b);}
DECL V4 OP/ (const T &a, V4ARG b) {return V4(a / b.x, a / b.y, a / b.z, a / b.w);}

DECL V4& OP+= (V4& a, V4ARG b) {a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a;}
DECL V4& OP-= (V4& a, V4ARG b) {a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a;}
DECL V4& OP*= (V4& a, V4ARG b) {a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a;}
DECL V4& OP/= (V4& a, V4ARG b) {a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w; return a;}

DECL V4& OP+= (V4& a, const T &b) {a.x += b; a.y += b; a.z += b; a.w += b; return a;}
DECL V4& OP-= (V4& a, const T &b) {a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a;}
DECL V4& OP*= (V4& a, const T &b) {a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a;}
DECL V4& OP/= (V4& a, const T &b) {a.x /= b; a.y /= b; a.z /= b; a.w /= b; return a;}

DECL bool OP== (V4ARG a, V4ARG b) {return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;}
DECL bool OP!= (V4ARG a, V4ARG b) {return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;}
DECL bool OP<  (V4ARG a, V4ARG b)
{
	if (a.x != b.x) return a.x < b.x;
	if (a.y != b.y) return a.y < b.y;
	if (a.z != b.z) return a.z < b.z;
	if (a.w != b.w) return a.w < b.w;
	return false;
}

DECL V4 abs (V4ARG a) {return V4(fabs(a.x), fabs(a.y), fabs(a.z), fabs(a.w));}
DECL V4 rcp (V4ARG a) {return V4(rcp(a.x), rcp(a.y), rcp(a.z), rcp(a.w));}
DECL V4 sqrt (V4ARG a) {return V4(sqrt (a.x), sqrt (a.y), sqrt (a.z), sqrt (a.w));}
DECL V4 rsqrt(V4ARG a) {return V4(rsqrt(a.x), rsqrt(a.y), rsqrt(a.z), rsqrt(a.w));}

DECL T reduce_add(V4ARG a) {return a.x + a.y + a.z + a.w;}
DECL T reduce_mul(V4ARG a) {return a.x * a.y * a.z * a.w;}
DECL T reduce_min(V4ARG a) {return min(a.x, a.y, a.z, a.w);}
DECL T reduce_max(V4ARG a) {return max(a.x, a.y, a.z, a.w);}

DECL V4 min(V4ARG a, V4ARG b) {return V4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));}
DECL V4 max(V4ARG a, V4ARG b) {return V4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));}

DECL T dot (V4ARG a, V4ARG b) {return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;}
DECL T length(V4ARG a) {return sqrt(dot(a,a));}
DECL T distance (V4ARG a, V4ARG b) {return length(a-b);}
DECL V4 normalize(V4ARG a) {return a*rsqrt(dot(a,a));}
DECL V4 cross(V4ARG a, V4ARG b)
{
	return V4(a.y*b.z - a.z*b.y,
		a.z*b.x - a.x*b.z,
		a.x*b.y - a.y*b.x,
		0.0f);
}

DECL V4 select( V4ARG s, V4ARG t, V4ARG f)
{
	return V4(select(s.x,t.x,f.x),
		select(s.y,t.y,f.y),
		select(s.z,t.z,f.z),
		select(s.w,t.w,f.w));
}
DECL V4 shuffle( V4ARG a, const Vec4<int>& mask ) { return V4( a[mask.x], a[mask.y], a[mask.z], a[mask.w] ); }

DECL bool any( V4ARG a ) { return a.x || a.y || a.z || a.w; }
DECL bool all( V4ARG a ) { return a.x && a.y && a.z && a.w; }

DECL int maxDim( V4ARG a ) 
  { 
    if (a.x > a.y)
	{
      if (a.x > a.z)
		  return 0;
	  else
		  return 2;
    }
	else
	{
		if (a.y > a.z)
			return 1;
		else
			return 2;
    }
  }

//////////////////////////////////////////////////////////////////////////////
// Helper / Util
//////////////////////////////////////////////////////////////////////////////

DECL V4 createPlane(const V4& a, const V4& b, const V4& c)
{
	const V4 edgeAB = b - a;
	const V4 edgeAC = c - a;

	V4 n = cross(edgeAB, edgeAC);
	n = normalize(n);
	n.w = -dot(n, a);

	return n;
}

DECL V4 convertToHSV(const V4& RGB)
{
	V4 HSV = V4(0.0f, 0.0f, 0.0f, 1.0f);  
	
	float minVal = min(RGB.x, min(RGB.y, RGB.z));  
	float maxVal = max(RGB.x, max(RGB.y, RGB.z));
	
	float delta = maxVal - minVal;
	
	if(delta != 0.0f)  
	{
		HSV.y = delta / maxVal;
		
		V4 delRGB = (((V4(maxVal) - RGB)/6.0f) + V4(delta/2.0f))/delta;
		
		if( RGB.x == maxVal )
			HSV.x = delRGB.z - delRGB.y;
		else if(RGB.y == maxVal)
			HSV.x = (1.0f/3.0f) + delRGB.x - delRGB.z;
		else if(RGB.z == maxVal)
			HSV.x = (2.0f/3.0f) + delRGB.y - delRGB.x;
		
		if (HSV.x < 0.0f)  
			HSV.x += 1.0f;
		if (HSV.x > 1.0f)
			HSV.x -= 1.0f;
	}
	
	HSV.z = maxVal;
	HSV.w = RGB.w;

	return HSV;
}

DECL void clipConvexPolygonAgainstPlane(const V4& plane, const std::vector<V4>& convexPoly, std::vector<V4>& convexPolyInside)
{
	convexPolyInside.clear();

	const int polygonInterior =  1;
    const int polygonBoundary =  0;
    const int polygonExterior = -1;
	const float boundaryEpsilon = 1.0e-3F;

	size_t negative = 0;
	size_t positive = 0;
    
	const size_t vertexCount = convexPoly.size();
	std::vector<int> location(vertexCount, 0);

    for(size_t a = 0; a < vertexCount; a++)
    {
        float d = dot(plane, convexPoly[a]);

		if(d < -boundaryEpsilon)
        {
			location[a] = polygonInterior;
			negative++;
		}

        if(d > boundaryEpsilon)
        {
            location[a] = polygonExterior;
            positive++;
        }
    }
    
    if(negative == 0)
    {
		convexPolyInside = convexPoly;
        return;
    }

    if(positive == 0)
    {
        return;
    }

    size_t previous = vertexCount - 1;
    for(size_t index = 0; index < vertexCount; ++index)
    {
        size_t loc = location[index];
        if(loc == polygonExterior)
        {
            if (location[previous] == polygonInterior)
            {
                const V4& vI = convexPoly[previous];
                const V4& vO = convexPoly[index];
                const V4& dIO = vO - vI;
                
                float t = - dot(plane, vI) / dot(plane, dIO);
				convexPolyInside.push_back( vI + dIO * t );
            }
        }
        else
        {
            const V4& vI = convexPoly[index];
            if ((loc == polygonInterior) && (location[previous] == polygonExterior))
            {
                const V4& vO = convexPoly[previous];
                const V4& dIO = vO - vI;
                
                float t = - dot(plane, vI) / dot(plane, dIO);
				convexPolyInside.push_back( vI + dIO * t );
            }
            
            convexPolyInside.push_back( vI );
        }
        
        previous = index;
    }
}

template<typename T> struct Vec_COMPX
{
    inline bool operator() (const T& a, const T& b) const
    {
        return (a.x < b.x);
    }
};
template<typename T> struct Vec_COMPY
{
    inline bool operator() (const T& a, const T& b) const
    {
        return (a.y < b.y);
    }
};
template<typename T> struct Vec_COMPZ
{
    inline bool operator() (const T& a, const T& b) const
    {
        return (a.z < b.z);
    }
};
template<typename T> struct Vec_COMPW
{
    inline bool operator() (const T& a, const T& b) const
    {
        return (a.w < b.w);
    }
};

//////////////////////////////////////////////////////////////////////////////
// Commonly used vector types
//////////////////////////////////////////////////////////////////////////////

typedef Vec1<bool >  Vec1b;
typedef Vec1<int>    Vec1i;
typedef Vec1<uchar> Vec1uc;
typedef Vec1<uint>  Vec1ui;
typedef Vec1<float>  Vec1f;
typedef Vec1<double> Vec1d;

typedef Vec2<bool >  Vec2b;
typedef Vec2<int>    Vec2i;
typedef Vec2<uchar> Vec2uc;
typedef Vec2<uint>  Vec2ui;
typedef Vec2<float>  Vec2f;
typedef Vec2<double> Vec2d;

typedef Vec4<bool >  Vec4b;
typedef Vec4<int>    Vec4i;
typedef Vec4<uchar> Vec4uc;
typedef Vec4<uint>  Vec4ui;
typedef Vec4<float>  Vec4f;
typedef Vec4<double> Vec4d;

#undef OP
#undef DECL
#undef V1
#undef V2
#undef V4
#undef V2ARG
#undef V4ARG

#endif
