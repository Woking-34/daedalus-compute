#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "platform.h"

#ifndef nullptr
	#define nullptr NULL
#endif

#ifdef True
	#undef True
#endif

#ifdef False
	#undef False
#endif

static struct NullTy {
} null;

static struct TrueTy {
	INLINE operator bool( ) const { return true; }
} True;

static struct FalseTy {
	INLINE operator bool( ) const { return false; }
} False;

static struct ZeroTy
{
	INLINE operator          double   ( ) const { return 0.0; }
	INLINE operator          float    ( ) const { return 0.0f; }
	INLINE operator          long long( ) const { return 0; }
	INLINE operator unsigned long long( ) const { return 0; }
	INLINE operator          long     ( ) const { return 0; }
	INLINE operator unsigned long     ( ) const { return 0; }
	INLINE operator          int      ( ) const { return 0; }
	INLINE operator unsigned int      ( ) const { return 0; }
	INLINE operator          short    ( ) const { return 0; }
	INLINE operator unsigned short    ( ) const { return 0; }
	INLINE operator          char     ( ) const { return 0; }
	INLINE operator unsigned char     ( ) const { return 0; }
} zero;

static struct OneTy
{
	INLINE operator          double   ( ) const { return 1.0; }
	INLINE operator          float    ( ) const { return 1.0f; }
	INLINE operator          long long( ) const { return 1; }
	INLINE operator unsigned long long( ) const { return 1; }
	INLINE operator          long     ( ) const { return 1; }
	INLINE operator unsigned long     ( ) const { return 1; }
	INLINE operator          int      ( ) const { return 1; }
	INLINE operator unsigned int      ( ) const { return 1; }
	INLINE operator          short    ( ) const { return 1; }
	INLINE operator unsigned short    ( ) const { return 1; }
	INLINE operator          char     ( ) const { return 1; }
	INLINE operator unsigned char     ( ) const { return 1; }
} one;

static struct NegInfTy
{
	INLINE operator          double   ( ) const { return -std::numeric_limits<double>::infinity(); }
	INLINE operator          float    ( ) const { return -std::numeric_limits<float>::infinity(); }
	INLINE operator          long long( ) const { return std::numeric_limits<long long>::min(); }
	INLINE operator unsigned long long( ) const { return std::numeric_limits<unsigned long long>::min(); }
	INLINE operator          long     ( ) const { return std::numeric_limits<long>::min(); }
	INLINE operator unsigned long     ( ) const { return std::numeric_limits<unsigned long>::min(); }
	INLINE operator          int      ( ) const { return std::numeric_limits<int>::min(); }
	INLINE operator unsigned int      ( ) const { return std::numeric_limits<unsigned int>::min(); }
	INLINE operator          short    ( ) const { return std::numeric_limits<short>::min(); }
	INLINE operator unsigned short    ( ) const { return std::numeric_limits<unsigned short>::min(); }
	INLINE operator          char     ( ) const { return std::numeric_limits<char>::min(); }
	INLINE operator unsigned char     ( ) const { return std::numeric_limits<unsigned char>::min(); }

} neg_inf;

static struct PosInfTy
{
	INLINE operator          double   ( ) const { return std::numeric_limits<double>::infinity(); }
	INLINE operator          float    ( ) const { return std::numeric_limits<float>::infinity(); }
	INLINE operator          long long( ) const { return std::numeric_limits<long long>::max(); }
	INLINE operator unsigned long long( ) const { return std::numeric_limits<unsigned long long>::max(); }
	INLINE operator          long     ( ) const { return std::numeric_limits<long>::max(); }
	INLINE operator unsigned long     ( ) const { return std::numeric_limits<unsigned long>::max(); }
	INLINE operator          int      ( ) const { return std::numeric_limits<int>::max(); }
	INLINE operator unsigned int      ( ) const { return std::numeric_limits<unsigned int>::max(); }
	INLINE operator          short    ( ) const { return std::numeric_limits<short>::max(); }
	INLINE operator unsigned short    ( ) const { return std::numeric_limits<unsigned short>::max(); }
	INLINE operator          char     ( ) const { return std::numeric_limits<char>::max(); }
	INLINE operator unsigned char     ( ) const { return std::numeric_limits<unsigned char>::max(); }
} inf, pos_inf;
/*
static struct NaNTy
{
	INLINE operator double( ) const { return std::numeric_limits<double>::quiet_NaN(); }
	INLINE operator float ( ) const { return std::numeric_limits<float>::quiet_NaN(); }
} nan;

static struct UlpTy
{
	INLINE operator double( ) const { return std::numeric_limits<double>::epsilon(); }
	INLINE operator float ( ) const { return std::numeric_limits<float>::epsilon(); }
} ulp;
*/
static struct PiTy
{
	INLINE operator double( ) const { return 3.14159265358979323846; }
	INLINE operator float ( ) const { return 3.14159265358979323846f; }
} pi;

static struct OneOverPiTy
{
	INLINE operator double( ) const { return 0.31830988618379069122; }
	INLINE operator float ( ) const { return 0.31830988618379069122f; }
} one_over_pi;

static struct TwoPiTy
{
	INLINE operator double( ) const { return 6.283185307179586232; }
	INLINE operator float ( ) const { return 6.283185307179586232f; }
} two_pi;

static struct OneOverTwoPiTy
{
	INLINE operator double( ) const { return 0.15915494309189534561; }
	INLINE operator float ( ) const { return 0.15915494309189534561f; }
} one_over_two_pi;

static struct FourPiTy
{
	INLINE operator double( ) const { return 12.566370614359172464; } 
	INLINE operator float ( ) const { return 12.566370614359172464f; }
} four_pi;

static struct OneOverFourPiTy
{
	INLINE operator double( ) const { return 0.079577471545947672804; }
	INLINE operator float ( ) const { return 0.079577471545947672804f; }
} one_over_four_pi;

#endif 
