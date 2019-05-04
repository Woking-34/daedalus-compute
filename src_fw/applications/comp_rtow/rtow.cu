#include <cuda.h>
#include <cuda_runtime_api.h>

#include <vector_types.h>
#include <vector_functions.h>

#define __CUDA_INTERNAL_COMPILATION__
#include <math_constants.h>
#include <math_functions.h>
#undef __CUDA_INTERNAL_COMPILATION__

#include <float.h>
#include <limits.h>

#include <math.h>

#include "cutil_math.h"

__forceinline__ __device__  float3 unit_vector(float3 v)
{
	return v / length(v);
}

__forceinline__ __device__ float schlick(float cosine, float ref_idx)
{
	float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
	r0 = r0 * r0;

	return r0 + (1.0f - r0)*pow((1.0f - cosine), 5.0f);

}

__forceinline__ __device__ float3 reflect_ex(float3 v, float3 n)
{
	return v - 2.0f*dot(v, n)*n;
}

__forceinline__ __device__ bool refract(const float3 v, const float3 n, float ni_over_nt, float3* refracted)
{
	float3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0f - ni_over_nt * ni_over_nt*(1 - dt * dt);

	if (discriminant > 0)
	{
		*refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
		return true;
	}
	else
	{
		return false;
	}
}

__forceinline__ __device__  float get_random(unsigned int *seed0, unsigned int *seed1)
{
	*seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);
	*seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	unsigned int ires = ((*seed0) << 16) + (*seed1);

	/* Convert to float */
	union {
		float f;
		unsigned int ui;
	} res;
	res.ui = (ires & 0x007fffff) | 0x40000000;

	return (res.f - 2.f) / 2.f;
}

__forceinline__ __device__  float3 random_in_unit_sphere(unsigned int *seed0, unsigned int *seed1)
{
	float px, py, pz;
	do {
		px = 2.0f*get_random(seed0, seed1) - 1.0f;
		py = 2.0f*get_random(seed0, seed1) - 1.0f;
		pz = 2.0f*get_random(seed0, seed1) - 1.0f;
	} while (px*px + py*py + pz*pz >= 1.0f);

	return make_float3(px, py, pz);
}

__forceinline__ __device__ float3 random_in_unit_disk(unsigned int *seed0, unsigned int *seed1)
{
	float px, py;
	do {
		px = 2.0f*get_random(seed0, seed1) - 1.0f;
		py = 2.0f*get_random(seed0, seed1) - 1.0f;
	} while (px*px + py*py >= 1.0f);

	return make_float3(px, py, 0.0f);
}

typedef struct
{
	float3 origin;
	float3 direction;
} ray;

__forceinline__ __device__  float3 point_at_parameter(const ray* r, float t) { return r->origin + t * r->direction; }

typedef struct
{
	float3 camera_origin;
	float camera_origin_w;
	float3 camera_lower_left_corner;
	float camera_lower_left_corner_w;
	float3 camera_horizontal;
	float camera_horizontal_w;
	float3 camera_vertical;
	float camera_vertical_w;
	float3 camera_u;
	float camera_u_w;
	float3 camera_v;
	float camera_v_w;
	float3 camera_w;
	float camera_w_w;

	float lens_radius;
	float pad0, pad1, pad2;
} camera;

__forceinline__ __device__ ray get_ray(const camera* cam, float s, float t)
{
	ray r;

	r.origin = cam->camera_origin;
	r.direction = cam->camera_lower_left_corner + s * cam->camera_horizontal + t * cam->camera_vertical - cam->camera_origin;

	return r;
}

__forceinline__ __device__ ray get_ray_dof(const camera* cam, float s, float t, unsigned int *seed0, unsigned int *seed1)
{
	float3 rd = cam->lens_radius * random_in_unit_disk(seed0, seed1);
	float3 offset = cam->camera_u * rd.x + cam->camera_v * rd.y;

	ray r;

	r.origin = cam->camera_origin + offset;
	r.direction = cam->camera_lower_left_corner + s * cam->camera_horizontal + t * cam->camera_vertical - cam->camera_origin - offset;

	return r;
}

typedef struct
{
	float t;
	float3 p;
	float3 normal;
	int mat_ptr;
} hit_record;

typedef struct
{
	float3 pos;
	float rad;
	int mat_ptr;
	int pad0, pad1, pad2;
} sphere;

typedef struct
{
	float3 albedo;
	float fuzz;
	int matType;
	int pad0, pad1, pad2;
} material;

__forceinline__ __device__ bool scatter(const material* __restrict__ mat, const ray* r_in, const hit_record* rec, float3* attenuation, ray* scattered, unsigned int* seed0, unsigned int* seed1)
{
	if (mat->matType == 0)
	{
		// lambertian
		float3 target = rec->p + rec->normal + random_in_unit_sphere(seed0, seed1);
		scattered->origin = rec->p;
		scattered->direction = target - rec->p;
		*attenuation = mat->albedo;
		return true;
	}
	else if (mat->matType == 1)
	{
		// metal
		float3 reflected = reflect_ex(unit_vector(r_in->direction), rec->normal);
		scattered->origin = rec->p;
		scattered->direction = reflected + mat->fuzz*random_in_unit_sphere(seed0, seed1);
		*attenuation = mat->albedo;
		return (dot(scattered->direction, rec->normal) > 0.0f);
	}
	else if (mat->matType == 2)
	{
		float3 outward_normal;
		float3 reflected = reflect_ex(r_in->direction, rec->normal);
		float ni_over_nt;
		*attenuation = make_float3(1.0f, 1.0f, 1.0f);
		float3 refracted;
		float reflect_prob;
		float cosine;

		if (dot(r_in->direction, rec->normal) > 0.0f)
		{
			outward_normal.x = -1.0f * (rec->normal.x);
			outward_normal.y = -1.0f * (rec->normal.y);
			outward_normal.z = -1.0f * (rec->normal.z);
			ni_over_nt = mat->fuzz;
			cosine = dot(r_in->direction, rec->normal) / length(r_in->direction);
			cosine = sqrt(1.0f - mat->fuzz*mat->fuzz*(1 - cosine * cosine));
		}
		else
		{
			outward_normal = rec->normal;
			ni_over_nt = 1.0f / mat->fuzz;
			cosine = -dot(r_in->direction, rec->normal) / length(r_in->direction);
		}

		if (refract(r_in->direction, outward_normal, ni_over_nt, &refracted))
			reflect_prob = schlick(cosine, mat->fuzz);
		else
			reflect_prob = 1.0f;

		if (get_random(seed0, seed1) < reflect_prob)
		{
			scattered->origin = rec->p;
			scattered->direction = reflected;
		}
		else
		{
			scattered->origin = rec->p;
			scattered->direction = refracted;
		}

		return true;

	}

	return false;
}

__forceinline__ __device__ bool sphere_hit(const sphere* __restrict__ currSphere, const ray* r, float t_min, float t_max, hit_record* rec)
{
	float3 oc = r->origin - currSphere->pos;
	float a = dot(r->direction, r->direction);
	float b = dot(oc, r->direction);
	float c = dot(oc, oc) - currSphere->rad*currSphere->rad;
	float discriminant = b * b - a * c;
	if (discriminant > 0.0f)
	{
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec->t = temp;
			rec->p = point_at_parameter(r, rec->t);
			rec->normal = (rec->p - currSphere->pos) / currSphere->rad;
			rec->mat_ptr = currSphere->mat_ptr;
			return true;
		}

		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec->t = temp;
			rec->p = point_at_parameter(r, rec->t);
			rec->normal = (rec->p - currSphere->pos) / currSphere->rad;
			rec->mat_ptr = currSphere->mat_ptr;
			return true;
		}
	}

	return false;
}

__forceinline__ __device__ bool hitable_list_hit(const int sphereNum, const sphere* __restrict__ spheres, const ray* r, float t_min, float t_max, hit_record* rec)
{
	hit_record temp_rec;
	bool hit_anything = false;
	float closest_so_far = t_max;

	for (int i = 0; i < sphereNum; ++i)
	{
		if (sphere_hit(spheres + i, r, t_min, closest_so_far, &temp_rec))
		{
			hit_anything = true;
			closest_so_far = temp_rec.t;
			*rec = temp_rec;
		}
	}

	return hit_anything;
}

__forceinline__ __device__ float3 color(int sphereNum, const sphere* __restrict__ spheres, const material* __restrict__ materials, const ray* r, unsigned int* seed0, unsigned int* seed1)
{
	ray cur_ray;
	cur_ray.origin = r->origin;
	cur_ray.direction = r->direction;

	float3 cur_attenuation = make_float3(1.0f, 1.0f, 1.0f);

	for (int i = 0; i < 50; i++)
	{
		hit_record rec;
		if (hitable_list_hit(sphereNum, spheres, &cur_ray, 0.001f, FLT_MAX, &rec))
		{
			ray scattered;
			float3 attenuation;
			if (scatter(materials + rec.mat_ptr, &cur_ray, &rec, &attenuation, &scattered, seed0, seed1))
			{
				cur_attenuation *= attenuation;
				cur_ray.origin = scattered.origin;
				cur_ray.direction = scattered.direction;
			}
			else
			{
				return make_float3(0.0f, 0.0f, 0.0f);
			}
		}
		else
		{
			float3 unit_direction = unit_vector(cur_ray.direction);
			float t = 0.5f*(unit_direction.y + 1.0f);
			float3 c = (1.0f - t)*make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
			return cur_attenuation * c;
		}
	}

	return make_float3(0.0f, 0.0f, 0.0f); // exceeded recursion
}

extern "C"
__global__
void rtow(
	int max_x, int max_y, int sampleNum, int sphereNum,
	const sphere* __restrict__ spheres,
	const material* __restrict__ materials,
	const camera* __restrict__ gCamera,
	unsigned int* __restrict__ gSeed0,
	unsigned int* __restrict__ gSeed1,
	cudaSurfaceObject_t destSurf)
{
	// workitem/worksize info
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	uint j = blockIdx.y * blockDim.y + threadIdx.y;
	if ((i >= max_x) || (j >= max_y))
		return;

	unsigned int seed0 = gSeed0[j * max_x + i];
	unsigned int seed1 = gSeed1[j * max_x + i];

	camera cam = gCamera[0];

	float3 col = make_float3(0.0f, 0.0f, 0.0f);
	for (int s = 0; s < sampleNum; ++s)
	{
		float u = (float)(i + get_random(&seed0, &seed1)) / (float)(max_x);
		float v = (float)(j + get_random(&seed0, &seed1)) / (float)(max_y);

		//ray r = get_ray(&cam, u, v);
		ray r = get_ray_dof(&cam, u, v, &seed0, &seed1);
		col += color(sphereNum, spheres, materials, &r, &seed0, &seed1);
	}

	gSeed0[j * max_x + i] = seed0;
	gSeed1[j * max_x + i] = seed1;

	col /= (float)(sampleNum);
	col.x = sqrt(col.x);
	col.y = sqrt(col.y);
	col.z = sqrt(col.z);

	float4 finalColor = make_float4(col, 1.0f);
	surf2Dwrite(finalColor, destSurf, i * sizeof(float4), max_y - 1 - j);
}

extern "C"
void cuK_rtow
(
	int max_x, int max_y, int sampleNum, int sphereNum, void* cuMSpheres, void* cuMMaterials, void* cuMCamera, void* cuMSeed0, void* cuMSeed1, cudaSurfaceObject_t cuSurface,
	unsigned int gSizeX, unsigned int gSizeY, unsigned int lSizeX, unsigned int lSizeY
)
{
	dim3 block(lSizeX, lSizeY, 1);
	dim3 grid(gSizeX / block.x, gSizeY / block.y, 1);

	rtow << < grid, block >> >(max_x, max_y, sampleNum, sphereNum, (sphere*)cuMSpheres, (material*)cuMMaterials, (camera*)cuMCamera, (unsigned int*)cuMSeed0, (unsigned int*)cuMSeed1, cuSurface);
}