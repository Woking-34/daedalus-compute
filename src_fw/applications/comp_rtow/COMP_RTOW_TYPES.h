#ifndef COMP_RTOW_TYPES_H
#define COMP_RTOW_TYPES_H

struct rtow_hit_record
{
	float px, py, pz;
	float t;
	float nx, ny, nz;
	int mat_ptr;
};

struct rtow_sphere
{
	float posX, posY, posZ;
	float rad;
	int mat_ptr;
	int pad0, pad1, pad2;
};

struct rtow_material
{
	float albedoR, albedoG, albedoB;
	float fuzz;
	int matType;
	int pad0, pad1, pad2;
};

#endif