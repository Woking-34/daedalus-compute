inline float3 unit_vector(float3 v)
{
	return v / length(v);
}

inline float schlick(float cosine, float ref_idx)
{
	float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
	r0 = r0*r0;

	return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

inline float3 reflect_ex(float3 v, float3 n)
{
	return v - 2.0f*dot(v,n)*n;
}

inline bool refract(const float3 v, const float3 n, float ni_over_nt, float3* refracted)
{
	float3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);

	if (discriminant > 0)
	{
		*refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
		return true;
	}
	else
	{
		return false;
	}
}

inline float get_random(unsigned int *seed0, unsigned int *seed1)
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

inline float3 random_in_unit_sphere(unsigned int *seed0, unsigned int *seed1)
{
	float px,py,pz;
	
	do {
		px = 2.0f*get_random(seed0, seed1) - 1.0f;
		py = 2.0f*get_random(seed0, seed1) - 1.0f;
		pz = 2.0f*get_random(seed0, seed1) - 1.0f;
	} while (px*px + py*py + pz*pz >= 1.0f);
	
	return (float3)(px,py,pz);
}

inline float3 random_in_unit_disk(unsigned int *seed0, unsigned int *seed1)
{
	float px,py;
	
	do {
		px = 2.0f*get_random(seed0, seed1) - 1.0f;
		py = 2.0f*get_random(seed0, seed1) - 1.0f;
	} while (px*px + py*py >= 1.0f);
	
	return (float3)(px,py,0.0f);
}

typedef struct
{
	float3 origin;
	float3 direction;
} ray;

inline float3 point_at_parameter(const ray* r, float t) { return r->origin + t*r->direction; }

typedef struct
{
	float4 camera_origin;
	float4 camera_lower_left_corner;
	float4 camera_horizontal;
	float4 camera_vertical;
	float4 camera_u;
	float4 camera_v;
	float4 camera_w;
	
	float lens_radius;
	float pad0, pad1, pad2;
} camera;

inline ray get_ray(const camera* cam, float s, float t)
{
	ray r;

	r.origin = cam->camera_origin.xyz;
	r.direction = cam->camera_lower_left_corner.xyz + s*cam->camera_horizontal.xyz + t*cam->camera_vertical.xyz - cam->camera_origin.xyz;

	return r;
}

inline ray get_ray_dof(const camera* cam, float s, float t, unsigned int *seed0, unsigned int *seed1)
{
	float3 rd = cam->lens_radius * random_in_unit_disk(seed0, seed1);
	float3 offset = cam->camera_u.xyz * rd.x + cam->camera_v.xyz * rd.y;
	
	ray r;
	
	r.origin = cam->camera_origin.xyz + offset;
	r.direction = cam->camera_lower_left_corner.xyz + s*cam->camera_horizontal.xyz + t*cam->camera_vertical.xyz - cam->camera_origin.xyz - offset;
	
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
	float4 pos_rad;
	int mat_ptr;
	int pad0, pad1, pad2;
} sphere;

typedef struct 
{
	float4 albedo_fuzz;
	int matType;
	int pad0, pad1, pad2;
} material;

inline bool scatter(__global const material* mat, const ray* r_in, const hit_record* rec, float3* attenuation, ray* scattered, unsigned int* seed0, unsigned int* seed1)
{
	if(mat->matType == 0)
	{
		// lambertian
		float3 target = rec->p + rec->normal + random_in_unit_sphere(seed0, seed1);
		scattered->origin = rec->p;
		scattered->direction = target-rec->p;
		*attenuation = mat->albedo_fuzz.xyz;
		return true;
	}
	else if(mat->matType == 1)
	{
		// metal
		float3 reflected = reflect_ex(unit_vector(r_in->direction), rec->normal);
		scattered->origin = rec->p;
		scattered->direction = reflected + mat->albedo_fuzz.w*random_in_unit_sphere(seed0, seed1);
		*attenuation = mat->albedo_fuzz.xyz;
		return (dot(scattered->direction, rec->normal) > 0.0f);
	}
	else if(mat->matType == 2)
	{
		float3 outward_normal;
		float3 reflected = reflect_ex(r_in->direction, rec->normal);
		float ni_over_nt;
		*attenuation = (float3)(1.0f, 1.0f, 1.0f);
		float3 refracted;
		float reflect_prob;
		float cosine;

		if (dot(r_in->direction, rec->normal) > 0.0f)
		{
			outward_normal = -rec->normal;
			ni_over_nt = mat->albedo_fuzz.w;
			cosine = dot(r_in->direction, rec->normal) / length(r_in->direction);
			cosine = sqrt(1.0f - mat->albedo_fuzz.w*mat->albedo_fuzz.w*(1-cosine*cosine));
		}
		else
		{
			outward_normal = rec->normal;
			ni_over_nt = 1.0f / mat->albedo_fuzz.w;
			cosine = -dot(r_in->direction, rec->normal) / length(r_in->direction);
		}

		if (refract(r_in->direction, outward_normal, ni_over_nt, &refracted))
			reflect_prob = schlick(cosine, mat->albedo_fuzz.w);
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

inline bool sphere_hit(__global const sphere* currSphere, const ray* r, float t_min, float t_max, hit_record* rec)
{
	float3 oc = r->origin - currSphere->pos_rad.xyz;
	float a = dot(r->direction, r->direction);
	float b = dot(oc, r->direction);
	float c = dot(oc, oc) - currSphere->pos_rad.w*currSphere->pos_rad.w;
	float discriminant = b*b - a*c;
	if (discriminant > 0.0f)
	{
		float temp = (-b - sqrt(discriminant))/a;
		if (temp < t_max && temp > t_min) {
			rec->t = temp;
			rec->p = point_at_parameter(r, rec->t);
			rec->normal = (rec->p - currSphere->pos_rad.xyz) / currSphere->pos_rad.w;
			rec->mat_ptr = currSphere->mat_ptr;

			return true;
		}

		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec->t = temp;
			rec->p = point_at_parameter(r, rec->t);
			rec->normal = (rec->p - currSphere->pos_rad.xyz) / currSphere->pos_rad.w;
			rec->mat_ptr = currSphere->mat_ptr;

			return true;
		}
	}

	return false;
}

inline bool hitable_list_hit(const int sphereNum, __global const sphere* spheres, const ray* r, float t_min, float t_max, hit_record* rec)
{
	hit_record temp_rec;
	bool hit_anything = false;
	float closest_so_far = t_max;

	for (int i = 0; i < sphereNum; ++i)
	{
		if (sphere_hit(spheres+i, r, t_min, closest_so_far, &temp_rec))
		{
			hit_anything = true;
			closest_so_far = temp_rec.t;
			*rec = temp_rec;
		}
	}

	return hit_anything;
}

inline float3 color(int sphereNum, __global const sphere* spheres, __global const material* materials, const ray* r, unsigned int* seed0, unsigned int* seed1)
{
	ray cur_ray;
	cur_ray.origin = r->origin;
	cur_ray.direction = r->direction;

	float3 cur_attenuation = (float3)(1.0f,1.0f,1.0f);

	for(int i = 0; i < 50; i++)
	{
		hit_record rec;
		if (hitable_list_hit(sphereNum, spheres, &cur_ray, 0.001f, FLT_MAX, &rec))
		{
			ray scattered;
			float3 attenuation;
			if(scatter(materials+rec.mat_ptr, &cur_ray, &rec, &attenuation, &scattered, seed0, seed1))
			{
				cur_attenuation *= attenuation;
				cur_ray.origin = scattered.origin;
				cur_ray.direction = scattered.direction;
			}
			else
			{
				return (float3)(0.0f,0.0f,0.0f);
			}
		}
		else
		{
			float3 unit_direction = unit_vector(cur_ray.direction);
			float t = 0.5f*(unit_direction.y + 1.0f);
			float3 c = (1.0f-t)*(float3)(1.0f, 1.0f, 1.0f) + t*(float3)(0.5f, 0.7f, 1.0f);

			return cur_attenuation * c;
		}
	}

	return (float3)(0.0f,0.0f,0.0f); // exceeded recursion
}

__kernel
void render(
	int max_x, int max_y, int sampleNum, int sphereNum,
	__global const sphere* restrict spheres,
	__global const material* restrict materials,
	__global const camera* restrict gCamera,
	__global unsigned int* restrict gSeed0,
	__global unsigned int* restrict gSeed1,
	__write_only image2d_t destTex
)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	if((i >= max_x) || (j >= max_y))
		return;

	unsigned int seed0 = gSeed0[j * max_x + i];
	unsigned int seed1 = gSeed1[j * max_x + i];
	
	camera cam = gCamera[0];

	float3 col = (float3)(0.0f, 0.0f, 0.0f);
	for(int s=0; s < sampleNum; ++s)
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
	col = sqrt(col);

	float4 finalColor = (float4)(col, 1.0f);
	write_imagef(destTex, (int2)(i,max_y-1-j), finalColor);
}