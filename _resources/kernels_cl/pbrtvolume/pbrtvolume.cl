typedef struct
{
	float3 origin;
	float3 dir;
} Ray;

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
} Camera;

static inline void
generateRay
(
	const Camera* cam,
	float s, float t, Ray* ray
)
{
	ray->origin = cam->camera_origin.xyz;
	ray->dir = cam->camera_lower_left_corner.xyz + s*cam->camera_horizontal.xyz + t*cam->camera_vertical.xyz - cam->camera_origin.xyz;
}

static inline bool
Inside(float3 p, float3 pMin, float3 pMax)
{
    return (p.x >= pMin.x && p.x <= pMax.x &&
            p.y >= pMin.y && p.y <= pMax.y &&
            p.z >= pMin.z && p.z <= pMax.z);
}

static inline bool
IntersectP(const Ray* ray, float3 pMin, float3 pMax, float *hit0, float *hit1)
{
    float t0 = -1e30f, t1 = 1e30f;

    float3 tNear = (pMin - ray->origin) / ray->dir;
    float3 tFar  = (pMax - ray->origin) / ray->dir;
    if (tNear.x > tFar.x) {
        float tmp = tNear.x;
        tNear.x = tFar.x;
        tFar.x = tmp;
    }
    t0 = max(tNear.x, t0);
    t1 = min(tFar.x, t1);

    if (tNear.y > tFar.y) {
        float tmp = tNear.y;
        tNear.y = tFar.y;
        tFar.y = tmp;
    }
    t0 = max(tNear.y, t0);
    t1 = min(tFar.y, t1);

    if (tNear.z > tFar.z) {
        float tmp = tNear.z;
        tNear.z = tFar.z;
        tFar.z = tmp;
    }
    t0 = max(tNear.z, t0);
    t1 = min(tFar.z, t1);
    
    if (t0 <= t1) {
        *hit0 = t0;
        *hit1 = t1;
        return true;
    }
    else
        return false;
}

static inline float Lerp(float t, float a, float b)
{
    return (1.f - t) * a + t * b;
	//return mix(a, b, t);
}

//static inline int Clamp(int v, int low, int high) {
//    return std::min(std::max(v, low), high);
//}

static inline float D(int x, int y, int z, int3 nVoxels, __global const float* restrict density)
{
    x = clamp(x, 0, nVoxels.x-1);
    y = clamp(y, 0, nVoxels.y-1);
    z = clamp(z, 0, nVoxels.z-1);
    return density[z*nVoxels.x*nVoxels.y + y*nVoxels.x + x];
}

static inline float3 Offset(float3 p, float3 pMin, float3 pMax)
{
    return (float3)((p.x - pMin.x) / (pMax.x - pMin.x),
                    (p.y - pMin.y) / (pMax.y - pMin.y),
                    (p.z - pMin.z) / (pMax.z - pMin.z));
}

static inline float Density(float3 Pobj, float3 pMin, float3 pMax, 
                            int3 nVoxels, __global const float* restrict density)
{
    if (!Inside(Pobj, pMin, pMax)) 
        return 0;
    // Compute voxel coordinates and offsets for _Pobj_
    float3 vox = Offset(Pobj, pMin, pMax);
    vox.x = vox.x * nVoxels.x - .5f;
    vox.y = vox.y * nVoxels.y - .5f;
    vox.z = vox.z * nVoxels.z - .5f;
    int vx = (int)(vox.x), vy = (int)(vox.y), vz = (int)(vox.z);
    float dx = vox.x - vx, dy = vox.y - vy, dz = vox.z - vz;

    // Trilinearly interpolate density values to compute local density
    float d00 = Lerp(dx, D(vx, vy, vz, nVoxels, density),     
                         D(vx+1, vy, vz, nVoxels, density));
    float d10 = Lerp(dx, D(vx, vy+1, vz, nVoxels, density),   
                         D(vx+1, vy+1, vz, nVoxels, density));
    float d01 = Lerp(dx, D(vx, vy, vz+1, nVoxels, density),   
                         D(vx+1, vy, vz+1, nVoxels, density));
    float d11 = Lerp(dx, D(vx, vy+1, vz+1, nVoxels, density), 
                         D(vx+1, vy+1, vz+1, nVoxels, density));
    float d0 = Lerp(dy, d00, d10);
    float d1 = Lerp(dy, d01, d11);
    return Lerp(dz, d0, d1);
}

static inline float transmittance
(
	float3 p0, float3 p1, float3 pMin,
	float3 pMax, float sigma_t, int3 nVoxels, __global const float* restrict density
)
{
    float rayT0, rayT1;
    Ray ray;
    ray.origin = p1;
    ray.dir = p0 - p1;

    // Find the parametric t range along the ray that is inside the volume.
    if (!IntersectP(&ray, pMin, pMax, &rayT0, &rayT1))
        return 1.;

    rayT0 = max(rayT0, 0.f);

    // Accumulate beam transmittance in tau
    float tau = 0;
    float rayLength = sqrt(ray.dir.x * ray.dir.x + ray.dir.y * ray.dir.y +
                           ray.dir.z * ray.dir.z);
    float stepDist = 0.2f;
    float stepT = stepDist / rayLength;

    float t = rayT0;
    float3 pos = ray.origin + ray.dir * rayT0;
    float3 dirStep = ray.dir * stepT;
    while (t < rayT1) {
        tau += stepDist * sigma_t * Density(pos, pMin, pMax, nVoxels, density);
        pos = pos + dirStep;
        t += stepT;
    }

    return exp(-tau);
}

static inline float distanceSquared(float3 a, float3 b) {
    float3 d = a-b;
    return d.x*d.x + d.y*d.y + d.z*d.z;
}

static inline float raymarch
(
	const Ray* ray,
	int3 nVoxels, __global const float* restrict density
)
{
    float rayT0, rayT1;
    float3 pMin = (float3)(.3f, -.2f, .3f);
	float3 pMax = (float3)(1.8f, 2.3f, 1.8f);
    float3 lightPos = (float3)(-1.f, 4.f, 1.5f);

    if (!IntersectP(ray, pMin, pMax, &rayT0, &rayT1))
        return 0.;

    rayT0 = max(rayT0, 0.f);

    // Parameters that define the volume scattering characteristics and
    // sampling rate for raymarching
    float Le = .25f;           // Emission coefficient
    float sigma_a = 10;        // Absorption coefficient
    float sigma_s = 10;        // Scattering coefficient
    float stepDist = 0.025f;   // Ray step amount
    float lightIntensity = 40; // Light source intensity

    float tau = 0.f;  // accumulated beam transmittance
    float L = 0;      // radiance along the ray
    float rayLength = sqrt(ray->dir.x * ray->dir.x + ray->dir.y * ray->dir.y +
                           ray->dir.z * ray->dir.z);
    float stepT = stepDist / rayLength;

    float t = rayT0;
    float3 pos = ray->origin + ray->dir * rayT0;
    float3 dirStep = ray->dir * stepT;
    while (t < rayT1) {
        float d = Density(pos, pMin, pMax, nVoxels, density);

        // terminate once attenuation is high
        float atten = exp(-tau);
        if (atten < .005f)
            break;

        // direct lighting
        float Li = lightIntensity / distanceSquared(lightPos, pos) * 
            transmittance(lightPos, pos, pMin, pMax, sigma_a + sigma_s,
                          nVoxels, density);
        L += stepDist * atten * d * sigma_s * (Li + Le);

        // update beam transmittance
        tau += stepDist * (sigma_a + sigma_s) * d;

        pos = pos + dirStep;
        t += stepT;
    }

    // Gamma correction
    return pow(L, 1.f / 2.2f);
}

__kernel
void render
(
	int max_x, int max_y, int3 nVoxels,
	__global const float* restrict density,
	__global const Camera* restrict gCamera,
	__write_only image2d_t destTex
)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	if((x >= max_x) || (y >= max_y))
		return;
	
	Camera cam = gCamera[0];
	float u = (float)(x) / (float)(max_x);
	float v = (float)(y) / (float)(max_y);
	
	Ray ray;
	generateRay(&cam, u, v, &ray);
	
	float4 finalColor = raymarch(&ray, nVoxels, density);
	write_imagef(destTex, (int2)(x,max_y-1-y), min(finalColor, (float4)(1.0f,1.0f,1.0f,1.0f)));
}