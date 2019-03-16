#ifndef PLATFORM_H
#define PLATFORM_H

#if defined _WIN32
#	ifndef WIN32_LEAN_AND_MEAN
#		define WIN32_LEAN_AND_MEAN 1
#	endif
#	ifndef NOMINMAX
#		define NOMINMAX
#	endif
#	ifndef _USE_MATH_DEFINES
#		define _USE_MATH_DEFINES
#	endif

	#include <windows.h>

#	ifdef min
#		undef min
#	endif
#	ifdef max
#		undef max
#	endif

#	ifdef near
#		undef near
#	endif
#	ifdef far
#		undef far
#	endif
#else
	#include <unistd.h>
	#include <sys/time.h>
	#include <sys/types.h>
	#include <sys/stat.h>
	#include <sys/param.h>
	#include <sys/sysctl.h>
#endif

#include <cmath>
#include <cfloat>
#include <limits>

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <memory>

#include <string>
#include <cstring>
#include <fstream>
#include <sstream>

#include <iomanip>
#include <iostream>

#include <map>
#include <list>
#include <vector>
#include <iterator>
#include <algorithm>

#define STRING(x) #x
#define PRINT(x) std::cout << STRING(x) << " = " << std::endl << (x) << std::endl

#ifndef nullptr
    #define nullptr 0
#endif

#define DEALLOC(pointer) if ((pointer) != nullptr) { delete (pointer); (pointer) = nullptr; }
#define DEALLOC_ARR(pointer) if ((pointer) != nullptr) { delete[] (pointer); (pointer) = nullptr; }

////////////////////////////////////////////////////////////////////////////////
/// Compiler
////////////////////////////////////////////////////////////////////////////////

/*! Visual C compiler */
#ifdef _MSC_VER
	#define __MSVC__
#endif

/*! GCC compiler */
#ifdef __GNUC__
	//#define __GNUC__
#endif

/*! Intel compiler */
#ifdef __INTEL_COMPILER
	#define __ICC__
#endif

#ifdef __MSVC__
	#define INLINE			__forceinline
	#define RESTRICT		__restrict
#else
	#define INLINE			inline __attribute__((always_inline))
	#define RESTRICT		__restrict
#endif

////////////////////////////////////////////////////////////////////////////////
/// Basic Types
////////////////////////////////////////////////////////////////////////////////

typedef unsigned char uchar;
typedef unsigned int uint;

#if defined(__MSVC__)
	typedef          __int64  int64;
	typedef unsigned __int64 uint64;
	typedef          __int32  int32;
	typedef unsigned __int32 uint32;
	typedef          __int16  int16;
	typedef unsigned __int16 uint16;
	typedef          __int8    int8;
	typedef unsigned __int8   uint8;
#else
	typedef          long long  int64;
	typedef unsigned long long uint64;
	typedef                int  int32;
	typedef unsigned       int uint32;
	typedef              short  int16;
	typedef unsigned     short uint16;
	typedef               char   int8;
	typedef unsigned      char  uint8;
#endif

#if defined(__MSVC__)
	#define isnan _isnan
	#define isinf(f) (!_finite((f)))
#endif

inline int cast_f2i(float f)
{
  union { float f; int i; } v; v.f = f; return v.i;
}
inline float cast_i2f(int i)
{
  union { float f; int i; } v; v.i = i; return v.f;
}

template<typename T> inline void endianswap(T *objp)
{
	unsigned char* memp = reinterpret_cast<unsigned char*>(objp);
	std::reverse(memp, memp + sizeof(T));
}

inline bool findFullPath(std::string& filePath)
{
	bool fileFound = false;
	const std::string resourcePath = "_resources/";

	filePath = resourcePath + filePath;
	for(unsigned int i = 0; i < 16; ++i)
	{
		std::ifstream file;
		file.open(filePath.c_str());
		if (file.is_open())
		{
			fileFound = true;
			break;
		}

		filePath = "../" + filePath;
	}

	return fileFound;
}

/** round n down to nearest multiple of m */
inline int roundDown(int n, int m)
{
	return n >= 0 ? (n / m) * m : ((n - m + 1) / m) * m;
}
 
/** round n up to nearest multiple of m */
inline int roundUp(int n, int m)
{
	return n >= 0 ? ((n + m - 1) / m) * m : (n / m) * m;
}

inline unsigned int roundDownPOW2(unsigned int n)
{
	int exponent=0;
	while(n>1)
	{
		++exponent;
		n>>=1;
	}
	return 1<<exponent;
}

inline unsigned int roundUpPOW2(unsigned int n)
{
	int exponent=0;
	--n;
	while(n)
	{
		++exponent;
		n>>=1;
	}
	return 1<<exponent;
}

inline unsigned int round_down_to_power_of_two(unsigned int n)
{
	int exponent=0;
	while(n>1)
	{
		++exponent;
		n>>=1;
	}
	return 1<<exponent;
}

// uchat buffer [0, 255]
void savePPM(const std::string& name, unsigned char* src, int width, int height, int numChannels);

// float buffer [0.0f, 1.0f]
void savePPM(const std::string& name, float* src, int width, int height, int numChannels);

// positions
void saveOBJ_pos(const std::string& name, unsigned int numVertices, unsigned int numTriangles, float* positionVec, unsigned int positionStride, unsigned int* indices = nullptr);

// positions+normals
void saveOBJ_pos_norm(const std::string& name, unsigned int numVertices, unsigned int numTriangles, float* positionVec, float* normalVec, unsigned int positionStride, unsigned int normalStride, unsigned int* indices = nullptr);
void saveOBJ_pos_norm(const std::string& name, unsigned int numVertices, unsigned int numTriangles, float* positionnormalVec, unsigned int positionStride, unsigned int normalStride, unsigned int* indices = nullptr);

inline float randomFLT()
{
  return (float)rand()/(float)RAND_MAX;
}

uchar* loadRaw(const std::string& fileName, size_t* fileSize = NULL);

void saveRaw(const std::string& fileName, size_t fileSize, void* data);

std::string loadStr(const std::string fileName);

std::string tolowerString(const std::string& str);
 
std::string trimString(const std::string& str, const std::string& whitespace = " \t\r\n");

std::string reduceString(const std::string& str, const std::string& fill = " ", const std::string& whitespace = " \t\r\n");

std::string formatInt(int i, int width, char pad);
std::string formatFloat(float f, int width, int precision, char pad);
std::string formatDouble(double d, int width, int precision, char pad);

template <typename T> inline std::string tabulateStrings(const std::string& info, const T& value, unsigned int length = 36, char f = ' ')
{
	std::stringstream ssValue;
	ssValue << value;

	std::stringstream ss;
	if(info.size() < length)
	{
		ss << std::setw(length-info.size() + ssValue.str().size()) << std::setfill(f) << value;
	}
	else
	{
		ss << f << value;
	}
	
	return info + ss.str();
}

#endif
