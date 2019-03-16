#ifndef OBB_H
#define OBB_H

#include "math/mathbase.h"
#include "math/core/vec.h"
#include "math/core/mat.h"

template<typename T> struct OBB
{
	Mat44<T> mat;	// local frame + center
	Vec4<T> halfextents;

	INLINE OBB() {}
};

#endif
