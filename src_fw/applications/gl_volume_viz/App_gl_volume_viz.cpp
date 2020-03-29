#include "App_gl_volume_viz.h"

#include "system/log.h"
#include "system/timer.h"
#include "system/filesystem.h"


//a2fVertexOffset lists the positions, relative to vertex0, of each of the 8 vertices of a cube
static const float vertOff[8][3] =
{
        {0.0, 0.0, 0.0},{1.0, 0.0, 0.0},{1.0, 1.0, 0.0},{0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},{1.0, 0.0, 1.0},{1.0, 1.0, 1.0},{0.0, 1.0, 1.0}
};

//a2iEdgeConnection lists the index of the endpoint vertices for each of the 12 edges of the cube
static const int edgeCon[12][2] = 
{
        {0,1}, {1,2}, {2,3}, {3,0},
        {4,5}, {5,6}, {6,7}, {7,4},
        {0,4}, {1,5}, {2,6}, {3,7}
};

//a2fEdgeDirection lists the direction vector (vertex1-vertex0) for each edge in the cube
static const float edgeDir[12][3] =
{
        {1.0, 0.0, 0.0},{0.0, 1.0, 0.0},{-1.0, 0.0, 0.0},{0.0, -1.0, 0.0},
        {1.0, 0.0, 0.0},{0.0, 1.0, 0.0},{-1.0, 0.0, 0.0},{0.0, -1.0, 0.0},
        {0.0, 0.0, 1.0},{0.0, 0.0, 1.0},{ 0.0, 0.0, 1.0},{0.0,  0.0, 1.0}
};

int cubeFlags[256]=
{
	0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 
	0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 
	0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 
	0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 
	0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60, 
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0, 
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, 
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460, 
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0, 
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230, 
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190, 
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

int triTable[256][16] =  
{
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

using namespace daedalus;

App AppInstance;

App::App()
{
	renderMode = 0;

	currWidth = 800;
	currHeight = 800;

	volumeTexId = 0;
	tffTexId = 0;

	cubeFlagsTexId = 0;
	triTableTexId = 0;

	volumeDataTFF = nullptr;

	if(1)
	{
		volumeName = "bucky.raw";

		volumeW = 32;
		volumeH = 32;
		volumeD = 32;

		volumeWPadded = 32;
		volumeHPadded = 32;
		volumeDPadded = 32;

		isoLevel = isoCurr = 0.3f;
	}

	if(0)
	{
		volumeName = "smoke.raw";

		volumeW = 80;
		volumeH = 80;
		volumeD = 40;

		volumeWPadded = 80;
		volumeHPadded = 80;
		volumeDPadded = 40;

		isoLevel = isoCurr = 0.3f;
	}

	if(0)
	{
		volumeName = "bonsai.raw";

		volumeW = 256;
		volumeH = 256;
		volumeD = 256;
	
		volumeWPadded = 256;
		volumeHPadded = 256;
		volumeDPadded = 256;
	
		isoLevel = isoCurr = 0.1f;
	}
	
	if(0)
	{
		volumeName = "skull.raw";

		volumeW = 256;
		volumeH = 256;
		volumeD = 256;
	
		volumeWPadded = 256;
		volumeHPadded = 256;
		volumeDPadded = 256;
	
		isoLevel = isoCurr = 0.175f;
	}
	
	if(0)
	{
		volumeName = "head256.raw";

		volumeW = 256;
		volumeH = 256;
		volumeD = 225;
	
		volumeWPadded = 256;
		volumeHPadded = 256;
		volumeDPadded = 256;
	
		isoLevel = isoCurr = 0.175f;
	}

	if(0)
	{
		volumeName = "brain.raw";

		volumeW = 181;
		volumeH = 217;
		volumeD = 181;
	
		volumeWPadded = 256;
		volumeHPadded = 256;
		volumeDPadded = 256;
	
		isoLevel = isoCurr = 0.125f;
	}
}	

App::~App()
{

}

void App::Initialize()
{
	bool hasIntegerTextureExt = hasGLExtension("GL_EXT_texture_integer");

	LOG_BOOL( hasIntegerTextureExt, "GL_EXT_texture_integer\n" );

	const Vec4f eye(1.75f, 1.75f, 1.75f, 1.0f);
	const Vec4f center(0.0f, 0.0f, 0.0f, 1.0f);
	const Vec4f up(0.0f, 1.0f, 0.0f, 0.0f);

	mainCamera.setViewParams( eye, center, up );
	mainCamera.setProjParams( 60.0f, (float)(currWidth)/(float)(currHeight), 0.15f, 100.0f );
	mainCamera.updateRays();

	appCamera = &mainCamera;

	albedoProgram.addFile("albedo_vertcolor.vert", GL_VERTEX_SHADER);
	albedoProgram.addFile("albedo_vertcolor.frag", GL_FRAGMENT_SHADER);
	albedoProgram.buildProgram();

	volumeProgramMC.addFile("marchingcubes.vert", GL_VERTEX_SHADER);
	volumeProgramMC.addFile("marchingcubes.geom", GL_GEOMETRY_SHADER);
	volumeProgramMC.addFile("marchingcubes.frag", GL_FRAGMENT_SHADER);
	volumeProgramMC.buildProgram();

	volumeProgramRC.addFile("volumeraycast.vert", GL_VERTEX_SHADER);
	volumeProgramRC.addFile("volumeraycast.frag", GL_FRAGMENT_SHADER);
	volumeProgramRC.buildProgram();

	volumeProgramRCTFF.addFile("volumeraycast.vert", GL_VERTEX_SHADER);
	volumeProgramRCTFF.addFile("volumeraycast_tff.frag", GL_FRAGMENT_SHADER);
	volumeProgramRCTFF.buildProgram();

	CHECK_GL;

	size_t tffSize = 0;
	volumeDataTFF = (Vec4uc*)loadRaw(FileSystem::GetRawFolder() + "tff.dat", &tffSize);

	volumeData.setName(volumeName);
	volumeData.setSize(1, volumeW, volumeH, volumeD);
	volumeData.loadFromFile();
	volumeData.scaleFloat(255.0f);

	{
		actualW = volumeData.getW();
 		actualH = volumeData.getH();
 		actualD = volumeData.getD();

 		const GLvoid* data = volumeData.getData();
		const GLvoid* dataFLT = volumeData.getDataFLT();

		{
			glGenTextures(1, &tffTexId);
			glBindTexture(GL_TEXTURE_1D, tffTexId);

			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

			glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, volumeDataTFF);

			glBindTexture(GL_TEXTURE_1D, 0);

			CHECK_GL;
		}

		{
			glGenTextures(1, &volumeTexId);
			glBindTexture(GL_TEXTURE_3D, volumeTexId);

			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
			
			glTexImage3D(GL_TEXTURE_3D, 0, GL_INTENSITY, actualW, actualH, actualD, 0, GL_LUMINANCE, GL_FLOAT, dataFLT);
			//glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, actualW, actualH, actualD, 0, GL_RED, GL_FLOAT, dataFLT);

			glBindTexture(GL_TEXTURE_3D, 0);

			CHECK_GL;
		}
	}

	{
		glActiveTexture(GL_TEXTURE1);
		glEnable(GL_TEXTURE_2D);

		glGenTextures(1, &cubeFlagsTexId);
		glBindTexture(GL_TEXTURE_2D, cubeFlagsTexId);
		
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, 256, 1, 0, GL_RED_INTEGER, GL_INT, cubeFlags);

		glBindTexture(GL_TEXTURE_2D, 0);

		CHECK_GL;
	}

	{
		glActiveTexture(GL_TEXTURE2);
		glEnable(GL_TEXTURE_2D);

		glGenTextures(1, &triTableTexId);
		glBindTexture(GL_TEXTURE_2D, triTableTexId);
		
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

		glTexImage2D( GL_TEXTURE_2D, 0, GL_R32I, 16, 256, 0, GL_RED_INTEGER, GL_INT, triTable);

		glBindTexture(GL_TEXTURE_2D, 0);

		CHECK_GL;
	}
	
	MeshFile myMeshFileFSQPos;
	myMeshFileFSQPos.createFSQPos();
	volumeMeshFSQRC.create( myMeshFileFSQPos );
	CHECK_GL;

	MeshFile myMeshFileBox;
	myMeshFileBox.createBoxWire(1.0f, 1.0f, 1.0f);
	myMeshFileBox.setColorVec(1.0f, 1.0f, 1.0f);
	myMeshFileBox.flattenIndices();
	debugBox.create( myMeshFileBox );
	CHECK_GL;

	MeshFile myMeshAxis;
	myMeshAxis.createWorldAxis(1.25f);
	debugAxis.create( myMeshAxis );
	CHECK_GL;

	MeshFile myMeshFileMCPoints;
	std::vector< Vec4f > mcpointsVec;

	for(float k=0; k<actualD; ++k)
	{
		for(float j=0; j<actualH; ++j)
		{
			for(float i=0; i<actualW; ++i)
			{
				mcpointsVec.push_back( Vec4f(i,j,k,1.0f) );
			}
		}
	}

	myMeshFileMCPoints.setPositionVec( &mcpointsVec[0], (unsigned int)mcpointsVec.size() );
	volumeMeshMCGS.create( myMeshFileMCPoints );
	CHECK_GL;

	// init gl state
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

	LOG( LogLine() << "Press F1-F4 to switch between different rendering modes!\n" );
}

void App::Update()
{
	if(specialKeyState[GLUT_KEY_F1])
	{
		specialKeyState[GLUT_KEY_F1] = false;
		renderMode = 0;
	}

	if(specialKeyState[GLUT_KEY_F2])
	{
		specialKeyState[GLUT_KEY_F2] = false;
		renderMode = 1;
	}

	if(specialKeyState[GLUT_KEY_F3])
	{
		specialKeyState[GLUT_KEY_F3] = false;
		renderMode = 2;
	}

	if(specialKeyState[GLUT_KEY_F4])
	{
		specialKeyState[GLUT_KEY_F4] = false;
		renderMode = 3;
	}

	if( normalKeyState['+'] )
	{
		//normalKeyState['+'] = false;
		isoLevel += 0.001f;
		isoLevel = clamp(isoLevel, 0.0f, 1.0f);

		//std::cout << isoLevel << std::endl;
	}

	if( normalKeyState['-'] )
	{
		//normalKeyState['-'] = false;
		isoLevel -= 0.001f;
		isoLevel = clamp(isoLevel, 0.0f, 1.0f);

		//std::cout << isoLevel << std::endl;
	}
}

void App::Render()
{
	mainCamera.setProjParams(60.0f, (float)(currWidth) / (float)(currHeight), 0.15f, 100.0f);
	mainCamera.updateRays();

	CHECK_GL;

	glViewport(0,0,currWidth,currHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	// render debug geometry
	if(0)
	{
		albedoProgram.useProgram();

		albedoProgram.setFloatMatrix44( createTranslationMatrix(0.5f, 0.5f, 0.5f) * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );
		debugBox.render( albedoProgram.getAttribLocations() );

		albedoProgram.setFloatMatrix44( Mat44f(one) * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );
		debugAxis.render( albedoProgram.getAttribLocations() );

		glUseProgram(0);

		CHECK_GL;
	}

	// render volume with marching cubes tri extraction
	if(renderMode == 0)
	{
		volumeProgramMC.useProgram();

		volumeProgramMC.setIntValue(0, "volumeTex");
		volumeProgramMC.setIntValue(1, "cubeFlagsTex");
		volumeProgramMC.setIntValue(2, "triTableTex");

		volumeProgramMC.setFloatMatrix44( Mat44f(one) * mainCamera.viewMat * mainCamera.projMat, "u_MVPMat" );

		volumeProgramMC.setFloatValue(isoLevel, "isolevel");
		volumeProgramMC.setFloatValue((float)(actualW), "volumeW");
		volumeProgramMC.setFloatValue((float)(actualH), "volumeH");
		volumeProgramMC.setFloatValue((float)(actualD), "volumeD");

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_3D, volumeTexId);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, cubeFlagsTexId);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, triTableTexId);

		volumeMeshMCGS.render( volumeProgramMC.getAttribLocations() );

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_3D, 0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, 0);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, 0);

		glUseProgram(0);

		CHECK_GL;
	}

	// render volume with raycast/raymarch
	if(renderMode == 1)
	{
		Mat44f mat = mainCamera.viewMat * mainCamera.projMat;
		mat = getMatrixtInverse(mat);
		
		Vec4f vec = mainCamera.getPosition();
		
		volumeProgramRC.useProgram();
		
		volumeProgramRC.setFloatMatrix44( mat, "u_VPInvMat" );
		volumeProgramRC.setFloatVector4( vec, "eyePosW" );
		volumeProgramRC.setIntValue(0, "volumeTex");
		
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_3D, volumeTexId);

		volumeMeshFSQRC.render( volumeProgramRC.getAttribLocations() );
		
		glBindTexture(GL_TEXTURE_3D, 0);

		glUseProgram(0);

		CHECK_GL;
	}

	// render volume with raycast/raymarch + transfer function
	if(renderMode == 2)
	{
		Mat44f mat = mainCamera.viewMat * mainCamera.projMat;
		mat = getMatrixtInverse(mat);
		
		Vec4f vec = mainCamera.getPosition();
		
		volumeProgramRCTFF.useProgram();
		
		volumeProgramRCTFF.setFloatMatrix44( mat, "u_VPInvMat" );
		volumeProgramRCTFF.setFloatVector4( vec, "eyePosW" );
		volumeProgramRCTFF.setIntValue(0, "volumeTex");
		volumeProgramRCTFF.setIntValue(1, "tffTex");
		
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_3D, volumeTexId);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_1D, tffTexId);

		volumeMeshFSQRC.render( volumeProgramRCTFF.getAttribLocations() );
		
		glBindTexture(GL_TEXTURE_3D, 0);
		glBindTexture(GL_TEXTURE_1D, 0);

		glUseProgram(0);

		CHECK_GL;
	}

	// render volume with axis aligned slices
	if(renderMode == 3)
	{
		glUseProgram(0);

		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf( mainCamera.projMat );
		
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf( mainCamera.viewMat );

		const int _Segs = 64;

		int _xSegs = _Segs;
		int _ySegs = _Segs;
		int _zSegs = _Segs;

		glColor4f(1.0f,1.0f,1.0f,1.0f);
		//glColor4f(1.0f,1.0f,1.0f,1.0f/16);
		
		glActiveTexture(GL_TEXTURE0);
		glEnable(GL_TEXTURE_3D);
		glBindTexture(GL_TEXTURE_3D, volumeTexId);

		glDisable(GL_CULL_FACE);
		glDepthMask(GL_FALSE);

		glEnable(GL_BLEND);
		
		glBlendEquation(GL_FUNC_ADD);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		
		//glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);
		
		Vec4f camDir = normalize( mainCamera.getDirection() );
		int camMaxDim = maxDim( abs( camDir ) ) + 1;
		//camMaxDim *= (int)sign( camDir[camMaxDim-1] );
		
		if(camMaxDim == 1)
		{
			for(int _x = 0; _x < _xSegs; ++_x)
			{
				float curr = (_x+1) / float(_xSegs+1);

				glBegin(GL_QUADS);
					glTexCoord3f(curr, 0.0f, 0.0f);
					glVertex3d(curr, 0.0f, 0.0f);

					glTexCoord3f(curr, 0.0f, 1.0f);
					glVertex3d(curr, 0.0f, 1.0f);

					glTexCoord3f(curr, 1.0f, 1.0f);
					glVertex3d(curr, 1.0f, 1.0f);

					glTexCoord3f(curr, 1.0f, 0.0f);
					glVertex3d(curr, 1.0f, 0.0f);
				glEnd();
			}
		}

		if(camMaxDim == -1)
		{
			for(int _x = _xSegs-1; _x >= 0; --_x)
			{
				float curr = (_x+1) / float(_xSegs+1);

				glBegin(GL_QUADS);
					glTexCoord3f(curr, 0.0f, 0.0f);
					glVertex3d(curr, 0.0f, 0.0f);

					glTexCoord3f(curr, 0.0f, 1.0f);
					glVertex3d(curr, 0.0f, 1.0f);

					glTexCoord3f(curr, 1.0f, 1.0f);
					glVertex3d(curr, 1.0f, 1.0f);

					glTexCoord3f(curr, 1.0f, 0.0f);
					glVertex3d(curr, 1.0f, 0.0f);
				glEnd();
			}
		}

		if(camMaxDim == 2)
		{
			for(int _y = 0; _y < _ySegs; ++_y)
			{
				float curr = (_y+1) / float(_ySegs+1);

				glBegin(GL_QUADS);
					glTexCoord3f(0.0f, curr, 0.0f);
					glVertex3d(0.0f, curr, 0.0f);

					glTexCoord3f(1.0f, curr, 0.0f);
					glVertex3d(1.0f, curr, 0.0f);

					glTexCoord3f(1.0f, curr, 1.0f);
					glVertex3d(1.0f, curr, 1.0f);

					glTexCoord3f(0.0f, curr, 1.0f);
					glVertex3d(0.0f, curr, 1.0f);
				glEnd();
			}
		}

		if(camMaxDim == -2)
		{
			for(int _y = _ySegs-1; _y >= 0; --_y)
			{
				float curr = (_y+1) / float(_ySegs+1);

				glBegin(GL_QUADS);
					glTexCoord3f(0.0f, curr, 0.0f);
					glVertex3d(0.0f, curr, 0.0f);

					glTexCoord3f(1.0f, curr, 0.0f);
					glVertex3d(1.0f, curr, 0.0f);

					glTexCoord3f(1.0f, curr, 1.0f);
					glVertex3d(1.0f, curr, 1.0f);

					glTexCoord3f(0.0f, curr, 1.0f);
					glVertex3d(0.0f, curr, 1.0f);
				glEnd();
			}
		}

		if(camMaxDim == 3)
		{
			for(int _z = 0; _z < _zSegs; ++_z)
			{
				float curr = (_z+1) / float(_zSegs+1);

				glBegin(GL_QUADS);
					glTexCoord3f(0.0f, 0.0f, curr);
					glVertex3d(0.0f, 0.0f, curr);

					glTexCoord3f(0.0f, 1.0f, curr);
					glVertex3d(0.0f, 1.0f, curr);

					glTexCoord3f(1.0f, 1.0f, curr);
					glVertex3d(1.0f, 1.0f, curr);

					glTexCoord3f(1.0f, 0.0f, curr);
					glVertex3d(1.0f, 0.0f, curr);
				glEnd();
			}
		}

		if(camMaxDim == -3)
		{
			for(int _z = _zSegs-1; _z >= 0; --_z)
			{
				float curr = (_z+1) / float(_zSegs+1);

				glBegin(GL_QUADS);
					glTexCoord3f(0.0f, 0.0f, curr);
					glVertex3d(0.0f, 0.0f, curr);

					glTexCoord3f(0.0f, 1.0f, curr);
					glVertex3d(0.0f, 1.0f, curr);

					glTexCoord3f(1.0f, 1.0f, curr);
					glVertex3d(1.0f, 1.0f, curr);

					glTexCoord3f(1.0f, 0.0f, curr);
					glVertex3d(1.0f, 0.0f, curr);
				glEnd();
			}
		}

		glDisable(GL_BLEND);

		glEnable(GL_CULL_FACE);
		glDepthMask(GL_TRUE);
	
		glBindTexture(GL_TEXTURE_3D, 0);
		glDisable(GL_TEXTURE_3D);
	}

	if(renderMode == 4)
	{
		glUseProgram(0);

		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf( mainCamera.projMat );
		
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf( mainCamera.viewMat );

		glDisable(GL_CULL_FACE);
		glDepthMask(GL_FALSE);

		glEnable(GL_BLEND);
		
		glBlendEquation(GL_FUNC_ADD);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		int gridSizeX = volumeData.getW();
 		int gridSizeY = volumeData.getH();
 		int gridSizeZ = volumeData.getD();

 		const GLvoid* data = volumeData.getData();
		const GLvoid* dataFLT = volumeData.getDataFLT();

		glScalef(1.0f/gridSizeX,1.0f/gridSizeY,1.0f/gridSizeZ);

		glBegin(GL_QUADS);
		for(int z=0, pos = 0; z<gridSizeZ; ++z)
		{
			for(int y=0; y<gridSizeY; ++y)
			{
				for(int x=0; x<gridSizeX; ++x, ++pos)
				{
					float f = 0.025f * ((float*)dataFLT)[pos];

					//if(f < 0.01f)
					//	continue;

					glColor4f(1.0f, 1.0f, 1.0f, f);
					//glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

					glVertex3i(x,   y, z);
					glVertex3i(x,   y+1, z);
					glVertex3i(x+1, y+1, z);
					glVertex3i(x+1, y, z);
				
					glVertex3i(x,   y, z+1);
					glVertex3i(x,   y+1, z+1);
					glVertex3i(x+1, y+1, z+1);
					glVertex3i(x+1, y, z+1);
				
					glVertex3i(x, y,   z);
					glVertex3i(x, y,   z+1);
					glVertex3i(x, y+1, z+1);
					glVertex3i(x, y+1, z);
				
					glVertex3i(x+1, y,   z);
					glVertex3i(x+1, y,   z+1);
					glVertex3i(x+1, y+1, z+1);
					glVertex3i(x+1, y+1, z);

					glVertex3i(x,   y, z);
					glVertex3i(x,   y, z+1);
					glVertex3i(x+1, y, z+1);
					glVertex3i(x+1, y, z);
				
					glVertex3i(x,   y+1, z);
					glVertex3i(x,   y+1, z+1);
					glVertex3i(x+1, y+1, z+1);
					glVertex3i(x+1, y+1, z);
				}
			}
		}
		glEnd();

		glDisable(GL_BLEND);

		glEnable(GL_CULL_FACE);
		glDepthMask(GL_TRUE);
	}

	CHECK_GL;
}

void App::Terminate()
{
	DEALLOC_ARR(volumeDataTFF);

	CHECK_GL;

	albedoProgram.clear();
	volumeProgramMC.clear();
	volumeProgramRC.clear();
	volumeProgramRCTFF.clear();

	volumeMeshMCGS.clear();
	volumeMeshFSQRC.clear();
	debugBox.clear();
	debugAxis.clear();

	// release GL
	if(volumeTexId)
	{
		glDeleteTextures(1, &volumeTexId);
		volumeTexId = 0;
	}

	if(volumeTexId)
	{
		glDeleteTextures(1, &tffTexId);
		tffTexId = 0;
	}

	if(cubeFlagsTexId)
	{
		glDeleteTextures(1, &cubeFlagsTexId);
		cubeFlagsTexId = 0;
	}

	if(triTableTexId)
	{
		glDeleteTextures(1, &triTableTexId);
		triTableTexId = 0;
	}

	CHECK_GL;
}

std::string App::GetName()
{
	return std::string("gl_volume_viz");
}