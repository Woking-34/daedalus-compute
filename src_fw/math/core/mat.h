#ifndef MATRIX_H
#define MATRIX_H

#include "math/mathbase.h"
#include "math/core/vec.h"

#define OP operator
#define DECL template <typename T> INLINE
#define MAT22 Mat22<T>
#define MAT44 Mat44<T>
#define MAT22ARG const Mat22<T>&
#define MAT44ARG const Mat44<T>&

////////////////////////////////////////////////////////////////////////////
// 4x4 Matrix (homogenous transformation)
////////////////////////////////////////////////////////////////////////////

template<typename T> struct Mat44
{
	T	m11, m12, m13, m14,
		m21, m22, m23, m24,
		m31, m32, m33, m34,
		m41, m42, m43, m44;

	INLINE Mat44() {}
	INLINE Mat44(const Mat44& m)
	{
		m11 = m.m11; m12 = m.m12; m13 = m.m13; m14 = m.m14;
		m21 = m.m21; m22 = m.m22; m23 = m.m23; m24 = m.m24;
		m31 = m.m31; m32 = m.m32; m33 = m.m33; m34 = m.m34;
		m41 = m.m41; m42 = m.m42; m43 = m.m43; m44 = m.m44;
	}
	INLINE Mat44(const T& x0, const T& y0, const T& z0, const T& w0, const T& x1, const T& y1, const T& z1, const T& w1, const T& x2, const T& y2, const T& z2, const T& w2, const T& x3, const T& y3, const T& z3, const T& w3)
	{
		m11 = x0; m12 = y0; m13 = z0; m14 = w0;
		m21 = x1; m22 = y1; m23 = z1; m24 = w1;
		m31 = x2; m32 = y2; m33 = z2; m34 = w2;
		m41 = x3; m42 = y3; m43 = z3; m44 = w3;
	}

	INLINE Mat44(ZeroTy)
	{
		m11 = T(zero); m12 = T(zero); m13 = T(zero); m14 = T(zero);
		m21 = T(zero); m22 = T(zero); m23 = T(zero); m24 = T(zero);
		m31 = T(zero); m32 = T(zero); m33 = T(zero); m34 = T(zero);
		m41 = T(zero); m42 = T(zero); m43 = T(zero); m44 = T(zero);
	}
	INLINE Mat44(OneTy)
	{
		m11 = T(one); m12 = T(zero); m13 = T(zero); m14 = T(zero);
		m21 = T(zero); m22 = T(one); m23 = T(zero); m24 = T(zero);
		m31 = T(zero); m32 = T(zero); m33 = T(one); m34 = T(zero);
		m41 = T(zero); m42 = T(zero); m43 = T(zero); m44 = T(one);
	}

	//INLINE const T& operator [](size_t elem) const { assert(elem < 16); return (&m11)[elem]; }
	//INLINE       T& operator [](size_t elem)       { assert(elem < 16); return (&m11)[elem]; }

	INLINE operator const T*() const { return (const T*)this; };
	INLINE operator       T*()       { return       (T*)this; };
};

DECL MAT44 OP+ (MAT44ARG a)
{
	return MAT44(+a.m11,+a.m12,+a.m13,+a.m14,
				 +a.m21,+a.m22,+a.m23,+a.m24,
				 +a.m31,+a.m32,+a.m33,+a.m34,
				 +a.m41,+a.m42,+a.m43,+a.m44);
}
DECL MAT44 OP- (MAT44ARG a)
{
	return MAT44(-a.m11,-a.m12,-a.m13,-a.m14,
				 -a.m21,-a.m22,-a.m23,-a.m24,
				 -a.m31,-a.m32,-a.m33,-a.m34,
				 -a.m41,-a.m42,-a.m43,-a.m44);
}

DECL MAT44 OP+ (MAT44ARG a, MAT44ARG b)
{
	return MAT44(a.m11+b.m11, a.m12+b.m12, a.m13+b.m13, a.m14+b.m14,
				 a.m21+b.m21, a.m22+b.m22, a.m23+b.m23, a.m24+b.m24,
				 a.m31+b.m31, a.m32+b.m32, a.m33+b.m33, a.m34+b.m34,
				 a.m41+b.m41, a.m42+b.m42, a.m43+b.m43, a.m44+b.m44);
}
DECL MAT44 OP- (MAT44ARG a, MAT44ARG b)
{
	return MAT44(a.m11-b.m11, a.m12-b.m12, a.m13-b.m13, a.m14-b.m14,
				 a.m21-b.m21, a.m22-b.m22, a.m23-b.m23, a.m24-b.m24,
				 a.m31-b.m31, a.m32-b.m32, a.m33-b.m33, a.m34-b.m34,
				 a.m41-b.m41, a.m42-b.m42, a.m43-b.m43, a.m44-b.m44);
}

DECL MAT44 OP* (MAT44ARG a, MAT44ARG b)
{
	MAT44 tmp;

	tmp[ 0] = (a[ 0] * b[ 0]) + (a[ 1] * b[ 4]) + (a[ 2] * b[ 8]) + (a[ 3] * b[12]);
	tmp[ 1] = (a[ 0] * b[ 1]) + (a[ 1] * b[ 5]) + (a[ 2] * b[ 9]) + (a[ 3] * b[13]);
	tmp[ 2] = (a[ 0] * b[ 2]) + (a[ 1] * b[ 6]) + (a[ 2] * b[10]) + (a[ 3] * b[14]);
	tmp[ 3] = (a[ 0] * b[ 3]) + (a[ 1] * b[ 7]) + (a[ 2] * b[11]) + (a[ 3] * b[15]);

	tmp[ 4] = (a[ 4] * b[ 0]) + (a[ 5] * b[ 4]) + (a[ 6] * b[ 8]) + (a[ 7] * b[12]);
	tmp[ 5] = (a[ 4] * b[ 1]) + (a[ 5] * b[ 5]) + (a[ 6] * b[ 9]) + (a[ 7] * b[13]);
	tmp[ 6] = (a[ 4] * b[ 2]) + (a[ 5] * b[ 6]) + (a[ 6] * b[10]) + (a[ 7] * b[14]);
	tmp[ 7] = (a[ 4] * b[ 3]) + (a[ 5] * b[ 7]) + (a[ 6] * b[11]) + (a[ 7] * b[15]);

	tmp[ 8] = (a[ 8] * b[ 0]) + (a[ 9] * b[ 4]) + (a[10] * b[ 8]) + (a[11] * b[12]);
	tmp[ 9] = (a[ 8] * b[ 1]) + (a[ 9] * b[ 5]) + (a[10] * b[ 9]) + (a[11] * b[13]);
	tmp[10] = (a[ 8] * b[ 2]) + (a[ 9] * b[ 6]) + (a[10] * b[10]) + (a[11] * b[14]);
	tmp[11] = (a[ 8] * b[ 3]) + (a[ 9] * b[ 7]) + (a[10] * b[11]) + (a[11] * b[15]);

	tmp[12] = (a[12] * b[ 0]) + (a[13] * b[ 4]) + (a[14] * b[ 8]) + (a[15] * b[12]);
	tmp[13] = (a[12] * b[ 1]) + (a[13] * b[ 5]) + (a[14] * b[ 9]) + (a[15] * b[13]);
	tmp[14] = (a[12] * b[ 2]) + (a[13] * b[ 6]) + (a[14] * b[10]) + (a[15] * b[14]);
	tmp[15] = (a[12] * b[ 3]) + (a[13] * b[ 7]) + (a[14] * b[11]) + (a[15] * b[15]);

	return tmp;
}
DECL MAT44 OP* (const T  a, MAT44ARG b)
{
	return MAT44(a*b.m11, a*b.m12, a*b.m13, a*b.m14,
				 a*b.m21, a*b.m22, a*b.m23, a*b.m24,
				 a*b.m31, a*b.m32, a*b.m33, a*b.m34,
				 a*b.m41, a*b.m42, a*b.m43, a*b.m44);
}
DECL MAT44 OP* (MAT44ARG a, const T  b)
{
	return MAT44(a.m11*b, a.m12*b, a.m13*b, a.m14*b,
				 a.m21*b, a.m22*b, a.m23*b, a.m24*b,
				 a.m31*b, a.m32*b, a.m33*b, a.m34*b,
				 a.m41*b, a.m42*b, a.m43*b, a.m44*b);
}

DECL MAT44 OP/ (MAT44ARG a, const T  b)
{
	const T divInv = 1.0f / b;

	return MAT44(a.m11*divInv, a.m12*divInv, a.m13*divInv, a.m14*divInv,
				 a.m21*divInv, a.m22*divInv, a.m23*divInv, a.m24*divInv,
				 a.m31*divInv, a.m32*divInv, a.m33*divInv, a.m34*divInv,
				 a.m41*divInv, a.m42*divInv, a.m43*divInv, a.m44*divInv);
}

DECL Vec4<T> OP* (const Vec4<T>& lhs, const Mat44<T>& rhs)
{
	return Vec4<T>(
        (lhs.x * rhs[ 0]) + (lhs.y * rhs[ 4]) + (lhs.z * rhs[ 8]) + (lhs.w * rhs[12]),
        (lhs.x * rhs[ 1]) + (lhs.y * rhs[ 5]) + (lhs.z * rhs[ 9]) + (lhs.w * rhs[13]),
        (lhs.x * rhs[ 2]) + (lhs.y * rhs[ 6]) + (lhs.z * rhs[10]) + (lhs.w * rhs[14]),
        (lhs.x * rhs[ 3]) + (lhs.y * rhs[ 7]) + (lhs.z * rhs[11]) + (lhs.w * rhs[15])
		);
}

DECL T getMatrixTrace(MAT44ARG mat)
{
	return mat.m11+mat.m22+mat.m33+mat.m44;
}

DECL T getMatrixDeterminant(MAT44ARG mat)
{
	return mat.m14 * mat.m23 * mat.m32 * mat.m41 - mat.m13 * mat.m24 * mat.m32 * mat.m41 - mat.m14 * mat.m22 * mat.m33 * mat.m41 + mat.m12 * mat.m24 * mat.m33 * mat.m41
		 + mat.m13 * mat.m22 * mat.m34 * mat.m41 - mat.m12 * mat.m23 * mat.m34 * mat.m41 - mat.m14 * mat.m23 * mat.m31 * mat.m42 + mat.m13 * mat.m24 * mat.m31 * mat.m42
		 + mat.m14 * mat.m21 * mat.m33 * mat.m42 - mat.m11 * mat.m24 * mat.m33 * mat.m42 - mat.m13 * mat.m21 * mat.m34 * mat.m42 + mat.m11 * mat.m23 * mat.m34 * mat.m42
		 + mat.m14 * mat.m22 * mat.m31 * mat.m43 - mat.m12 * mat.m24 * mat.m31 * mat.m43 - mat.m14 * mat.m21 * mat.m32 * mat.m43 + mat.m11 * mat.m24 * mat.m32 * mat.m43
		 + mat.m12 * mat.m21 * mat.m34 * mat.m43 - mat.m11 * mat.m22 * mat.m34 * mat.m43 - mat.m13 * mat.m22 * mat.m31 * mat.m44 + mat.m12 * mat.m23 * mat.m31 * mat.m44
		 + mat.m13 * mat.m21 * mat.m32 * mat.m44 - mat.m11 * mat.m23 * mat.m32 * mat.m44 - mat.m12 * mat.m21 * mat.m33 * mat.m44 + mat.m11 * mat.m22 * mat.m33 * mat.m44;
}

DECL MAT44 getMatrixTranspose(MAT44ARG mat)
{
	return Mat44<T>(mat.m11, mat.m21, mat.m31, mat.m41,
					mat.m12, mat.m22, mat.m32, mat.m42,
					mat.m13, mat.m23, mat.m33, mat.m43,
					mat.m14, mat.m24, mat.m34, mat.m44);
}

DECL MAT44 getMatrixtInverse(MAT44ARG mat)
{
	const T det = T(one) / getMatrixDeterminant(mat);

	return  Mat44<T>(
		det * (mat.m23*mat.m34*mat.m42 - mat.m24*mat.m33*mat.m42 + mat.m24*mat.m32*mat.m43 - mat.m22*mat.m34*mat.m43 - mat.m23*mat.m32*mat.m44 + mat.m22*mat.m33*mat.m44),
		det * (mat.m14*mat.m33*mat.m42 - mat.m13*mat.m34*mat.m42 - mat.m14*mat.m32*mat.m43 + mat.m12*mat.m34*mat.m43 + mat.m13*mat.m32*mat.m44 - mat.m12*mat.m33*mat.m44),
		det * (mat.m13*mat.m24*mat.m42 - mat.m14*mat.m23*mat.m42 + mat.m14*mat.m22*mat.m43 - mat.m12*mat.m24*mat.m43 - mat.m13*mat.m22*mat.m44 + mat.m12*mat.m23*mat.m44),
		det * (mat.m14*mat.m23*mat.m32 - mat.m13*mat.m24*mat.m32 - mat.m14*mat.m22*mat.m33 + mat.m12*mat.m24*mat.m33 + mat.m13*mat.m22*mat.m34 - mat.m12*mat.m23*mat.m34),

		det * (mat.m24*mat.m33*mat.m41 - mat.m23*mat.m34*mat.m41 - mat.m24*mat.m31*mat.m43 + mat.m21*mat.m34*mat.m43 + mat.m23*mat.m31*mat.m44 - mat.m21*mat.m33*mat.m44),
		det * (mat.m13*mat.m34*mat.m41 - mat.m14*mat.m33*mat.m41 + mat.m14*mat.m31*mat.m43 - mat.m11*mat.m34*mat.m43 - mat.m13*mat.m31*mat.m44 + mat.m11*mat.m33*mat.m44),
		det * (mat.m14*mat.m23*mat.m41 - mat.m13*mat.m24*mat.m41 - mat.m14*mat.m21*mat.m43 + mat.m11*mat.m24*mat.m43 + mat.m13*mat.m21*mat.m44 - mat.m11*mat.m23*mat.m44),
		det * (mat.m13*mat.m24*mat.m31 - mat.m14*mat.m23*mat.m31 + mat.m14*mat.m21*mat.m33 - mat.m11*mat.m24*mat.m33 - mat.m13*mat.m21*mat.m34 + mat.m11*mat.m23*mat.m34),

		det * (mat.m22*mat.m34*mat.m41 - mat.m24*mat.m32*mat.m41 + mat.m24*mat.m31*mat.m42 - mat.m21*mat.m34*mat.m42 - mat.m22*mat.m31*mat.m44 + mat.m21*mat.m32*mat.m44),
		det * (mat.m14*mat.m32*mat.m41 - mat.m12*mat.m34*mat.m41 - mat.m14*mat.m31*mat.m42 + mat.m11*mat.m34*mat.m42 + mat.m12*mat.m31*mat.m44 - mat.m11*mat.m32*mat.m44),
		det * (mat.m12*mat.m24*mat.m41 - mat.m14*mat.m22*mat.m41 + mat.m14*mat.m21*mat.m42 - mat.m11*mat.m24*mat.m42 - mat.m12*mat.m21*mat.m44 + mat.m11*mat.m22*mat.m44),
		det * (mat.m14*mat.m22*mat.m31 - mat.m12*mat.m24*mat.m31 - mat.m14*mat.m21*mat.m32 + mat.m11*mat.m24*mat.m32 + mat.m12*mat.m21*mat.m34 - mat.m11*mat.m22*mat.m34),

		det * (mat.m23*mat.m32*mat.m41 - mat.m22*mat.m33*mat.m41 - mat.m23*mat.m31*mat.m42 + mat.m21*mat.m33*mat.m42 + mat.m22*mat.m31*mat.m43 - mat.m21*mat.m32*mat.m43),
		det * (mat.m12*mat.m33*mat.m41 - mat.m13*mat.m32*mat.m41 + mat.m13*mat.m31*mat.m42 - mat.m11*mat.m33*mat.m42 - mat.m12*mat.m31*mat.m43 + mat.m11*mat.m32*mat.m43),
		det * (mat.m13*mat.m22*mat.m41 - mat.m12*mat.m23*mat.m41 - mat.m13*mat.m21*mat.m42 + mat.m11*mat.m23*mat.m42 + mat.m12*mat.m21*mat.m43 - mat.m11*mat.m22*mat.m43),
		det * (mat.m12*mat.m23*mat.m31 - mat.m13*mat.m22*mat.m31 + mat.m13*mat.m21*mat.m32 - mat.m11*mat.m23*mat.m32 - mat.m12*mat.m21*mat.m33 + mat.m11*mat.m22*mat.m33)
	);
}

DECL MAT44 createScaleMatrix(const T sx, const T sy, const T sz)
{
	return Mat44<T>(      sx, T(zero), T(zero), T(zero),
					 T(zero),     sy,  T(zero), T(zero),
					 T(zero), T(zero),      sz, T(zero),
					 T(zero), T(zero), T(zero),  T(one));
}

DECL MAT44 createScaleMatrix(const Vec4f& s)
{
	return Mat44<T>(     s.x, T(zero), T(zero), T(zero),
					 T(zero),    s.y,  T(zero), T(zero),
					 T(zero), T(zero),     s.z, T(zero),
					 T(zero), T(zero), T(zero),  T(one));
}

DECL MAT44 createRotationMatrix(const Vec4<T>& rAxis, const T rAngle)
{
	const Vec4<T> u = normalize(rAxis);
	const T s = sin(rAngle);
	const T c = cos(rAngle);

	return Mat44<T>(u.x*u.x+(1.0f-u.x*u.x)*c,   u.x*u.y*(1.0f-c)-u.z*s,   u.x*u.z*(1.0f-c)+u.y*s, T(zero),
					  u.x*u.y*(1.0f-c)+u.z*s, u.y*u.y+(1.0f-u.y*u.y)*c,   u.y*u.z*(1.0f-c)-u.x*s, T(zero),
					  u.x*u.z*(1.0f-c)-u.y*s,   u.y*u.z*(1.0f-c)+u.x*s, u.z*u.z+(1.0f-u.z*u.z)*c, T(zero),
									 T(zero),                  T(zero),                  T(zero),  T(one));
}

DECL MAT44 createRotationX(const T rAngle)
{
	const T s = sin(rAngle);
	const T c = cos(rAngle);

	return Mat44<T>( T(one), T(zero), T(zero), T(zero),
					T(zero),       c,       s, T(zero),
					T(zero),      -s,       c, T(zero),
					T(zero), T(zero), T(zero),  T(one));
}

DECL MAT44 createRotationY(const T rAngle)
{
	const T s = sin(rAngle);
	const T c = cos(rAngle);

	return Mat44<T>(      c, T(zero),      -s, T(zero),
					T(zero),  T(one), T(zero), T(zero),
					      s, T(zero),       c, T(zero),
					T(zero), T(zero), T(zero),  T(one));
}

DECL MAT44 createRotationZ(const T rAngle)
{
	const T s = sin(rAngle);
	const T c = cos(rAngle);

	return Mat44<T>(      c,       s, T(zero), T(zero),
					     -s,       c, T(zero), T(zero),
					T(zero), T(zero),  T(one), T(zero),
					T(zero), T(zero), T(zero),  T(one));
}

DECL MAT44 createTranslationMatrix(const T tx, const T ty, const T tz)
{
	return Mat44<T>( T(one), T(zero), T(zero), T(zero),
					T(zero),  T(one), T(zero), T(zero),
					T(zero), T(zero),  T(one), T(zero),
					     tx,      ty,      tz,  T(one));
}

DECL MAT44 createTranslationMatrix(const Vec4<T>& t)
{
	return createTranslationMatrix(t.x, t.y, t.z);
}

DECL MAT44 createViewLookAtMatrix(const Vec4<T>& _eye, const Vec4<T>& _center, const Vec4<T>& _up)
{
	Mat44<T> result;

	Vec4<T> forward = _center - _eye;
	forward = normalize(forward);

	Vec4<T> right = cross(forward, _up);
	right = normalize(right);

	Vec4<T> up = cross(right, forward);

	result[ 0] = right[0];
	result[ 4] = right[1];
	result[ 8] = right[2];

	result[ 1] = up[0];
	result[ 5] = up[1];
	result[ 9] = up[2];

	result[ 2] = -forward[0];
	result[ 6] = -forward[1];
	result[10] = -forward[2];

	result[12] = -dot(_eye, right);
	result[13] = -dot(_eye, up);
	result[14] = -dot(_eye, -forward);

	result[ 3] = T(zero);
	result[ 7] = T(zero);
	result[11] = T(zero);
	result[15] =  T(one);

	return result;
}

DECL MAT44 createProjectionPerspectiveMatrix(const T _left, const T _right, const T _bottom, const T _top, const T _near, const T _far)
{
	Mat44<T> result;

	result[ 0] = (2.0f*_near) / (_right - _left);
	result[ 1] = T(zero);
	result[ 2] = T(zero);
	result[ 3] = T(zero);

	result[ 4] = T(zero);
	result[ 5] = (2.0f*_near) / (_top-_bottom);
	result[ 6] = T(zero);
	result[ 7] = T(zero);

	result[ 8] = (_right+_left) / (_right - _left);
	result[ 9] = (_top+_bottom) / (_top - _bottom);
	result[10] = (_near + _far) / (_near - _far);
	result[11] = -T(one);

	result[12] = T(zero);
	result[13] = T(zero);
	result[14] = (2.0f * _near * _far) / (_near - _far);
	result[15] = T(zero);

	return result;
}

DECL MAT44 createProjectionPerspectiveMatrix(const T _fovY, const T _aspectRatio, const T _near, const T _far)
{
	Mat44<T> result;

	const T cotangent = T(one) / tan( deg2rad(_fovY*0.5f) );

	result[ 0] = cotangent / _aspectRatio;
	result[ 1] = T(zero);
	result[ 2] = T(zero);
	result[ 3] = T(zero);

	result[ 4] = T(zero);
	result[ 5] = cotangent;
	result[ 6] = T(zero);
	result[ 7] = T(zero);

	result[ 8] = T(zero);
	result[ 9] = T(zero);
	result[10] = (_near + _far) / (_near - _far);
	result[11] = -T(one);

	result[12] = T(zero);
	result[13] = T(zero);
	result[14] = (2.0f * _near * _far) / (_near - _far);
	result[15] = T(zero);

	return result;
}

typedef Mat44<float > Mat44f;
typedef Mat44<double> Mat44d;

#undef OP
#undef DECL
#undef MAT22
#undef MAT44
#undef MAT22ARG
#undef MAT44ARG

#endif
