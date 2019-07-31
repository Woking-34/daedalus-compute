#include "appfw/GLUTApplication.h"

#include "math/core/vec.h"
#include "math/core/mat.h"
#include "math/util/camera.h"

#include "assets/meshfile.h"
#include "assets/imagedata.h"
#include "assets/volumedata.h"

#include "glutil/glprogram.h"
#include "glutil/glmesh.h"
#include "glutil/glmesh_soa.h"

#include "include/NvFlex.h"
#include "include/NvFlexExt.h"
#include "include/NvFlexDevice.h"

#include "maths.h"
#include "perlin.h"

#include <utility>
#include <algorithm>
#include <vector>

// Flex Demo - DamBreak 5cm
// Flex Demo - Flag Cloth

std::vector<daedalus::Vec4f> initCol;
std::vector<daedalus::Vec4f> initPos;

int g_frame = 0;
int g_numSubsteps = 2;
bool g_profile = false;

bool g_pause = false;
bool g_step = false;
bool g_Error = false;

float g_dt = 1.0f / 60.0f;	// the time delta used for simulation
float g_realdt;				// the real world time delta between updates

int g_mouseParticle = -1;
float g_mouseT = 0.0f;
Vec3 g_mousePos;
float g_mouseMass;
bool g_mousePicked = false;

bool g_emit = false;
bool g_warmup = false;

float g_windTime = 0.0f;
float g_windFrequency = 0.1f;
float g_windStrength = 0.0f;

bool g_wavePool = false;
float g_waveTime = 0.0f;
float g_wavePlane;
float g_waveFrequency = 1.5f;
float g_waveAmplitude = 1.0f;
float g_waveFloorTilt = 0.0f;

float g_pointScale;
float g_ropeScale;
float g_drawPlaneBias;	// move planes along their normal for rendering

Vec3 g_sceneLower;
Vec3 g_sceneUpper;

int g_device = -1;
char g_deviceName[256];
bool g_extensions = true;

NvFlexSolver* g_solver;
NvFlexSolverDesc g_solverDesc;
NvFlexLibrary* g_flexLib;
NvFlexParams g_params;

int g_maxDiffuseParticles;
int g_maxNeighborsPerParticle;
int g_numExtraParticles;
int g_numExtraMultiplier = 1;
int g_maxContactsPerParticle;

struct SimBuffers
{
	NvFlexVector<Vec4> positions;
	NvFlexVector<Vec4> restPositions;
	NvFlexVector<Vec3> velocities;
	NvFlexVector<int> phases;
	NvFlexVector<float> densities;
	NvFlexVector<Vec4> anisotropy1;
	NvFlexVector<Vec4> anisotropy2;
	NvFlexVector<Vec4> anisotropy3;
	NvFlexVector<Vec4> normals;
	NvFlexVector<Vec4> smoothPositions;
	NvFlexVector<Vec4> diffusePositions;
	NvFlexVector<Vec4> diffuseVelocities;
	NvFlexVector<int> diffuseCount;

	NvFlexVector<int> activeIndices;

	// convexes
	NvFlexVector<NvFlexCollisionGeometry> shapeGeometry;
	NvFlexVector<Vec4> shapePositions;
	NvFlexVector<Quat> shapeRotations;
	NvFlexVector<Vec4> shapePrevPositions;
	NvFlexVector<Quat> shapePrevRotations;
	NvFlexVector<int> shapeFlags;

	// rigids
	NvFlexVector<int> rigidOffsets;
	NvFlexVector<int> rigidIndices;
	NvFlexVector<int> rigidMeshSize;
	NvFlexVector<float> rigidCoefficients;
	NvFlexVector<float> rigidPlasticThresholds;
	NvFlexVector<float> rigidPlasticCreeps;
	NvFlexVector<Quat> rigidRotations;
	NvFlexVector<Vec3> rigidTranslations;
	NvFlexVector<Vec3> rigidLocalPositions;
	NvFlexVector<Vec4> rigidLocalNormals;

	// inflatables
	NvFlexVector<int> inflatableTriOffsets;
	NvFlexVector<int> inflatableTriCounts;
	NvFlexVector<float> inflatableVolumes;
	NvFlexVector<float> inflatableCoefficients;
	NvFlexVector<float> inflatablePressures;

	// springs
	NvFlexVector<int> springIndices;
	NvFlexVector<float> springLengths;
	NvFlexVector<float> springStiffness;

	NvFlexVector<int> triangles;
	NvFlexVector<Vec3> triangleNormals;
	NvFlexVector<Vec3> uvs;

	SimBuffers(NvFlexLibrary* l) :
		positions(l), restPositions(l), velocities(l), phases(l), densities(l),
		anisotropy1(l), anisotropy2(l), anisotropy3(l), normals(l), smoothPositions(l),
		diffusePositions(l), diffuseVelocities(l), diffuseCount(l), activeIndices(l),
		shapeGeometry(l), shapePositions(l), shapeRotations(l), shapePrevPositions(l),
		shapePrevRotations(l), shapeFlags(l), rigidOffsets(l), rigidIndices(l), rigidMeshSize(l),
		rigidCoefficients(l), rigidPlasticThresholds(l), rigidPlasticCreeps(l), rigidRotations(l), rigidTranslations(l),
		rigidLocalPositions(l), rigidLocalNormals(l), inflatableTriOffsets(l),
		inflatableTriCounts(l), inflatableVolumes(l), inflatableCoefficients(l),
		inflatablePressures(l), springIndices(l), springLengths(l),
		springStiffness(l), triangles(l), triangleNormals(l), uvs(l)
	{}
};

SimBuffers* g_buffers;

void MapBuffers(SimBuffers* buffers)
{
	buffers->positions.map();
	buffers->restPositions.map();
	buffers->velocities.map();
	buffers->phases.map();
	buffers->densities.map();
	buffers->anisotropy1.map();
	buffers->anisotropy2.map();
	buffers->anisotropy3.map();
	buffers->normals.map();
	buffers->diffusePositions.map();
	buffers->diffuseVelocities.map();
	buffers->diffuseCount.map();
	buffers->smoothPositions.map();
	buffers->activeIndices.map();

	// convexes
	buffers->shapeGeometry.map();
	buffers->shapePositions.map();
	buffers->shapeRotations.map();
	buffers->shapePrevPositions.map();
	buffers->shapePrevRotations.map();
	buffers->shapeFlags.map();

	buffers->rigidOffsets.map();
	buffers->rigidIndices.map();
	buffers->rigidMeshSize.map();
	buffers->rigidCoefficients.map();
	buffers->rigidPlasticThresholds.map();
	buffers->rigidPlasticCreeps.map();
	buffers->rigidRotations.map();
	buffers->rigidTranslations.map();
	buffers->rigidLocalPositions.map();
	buffers->rigidLocalNormals.map();

	buffers->springIndices.map();
	buffers->springLengths.map();
	buffers->springStiffness.map();

	// inflatables
	buffers->inflatableTriOffsets.map();
	buffers->inflatableTriCounts.map();
	buffers->inflatableVolumes.map();
	buffers->inflatableCoefficients.map();
	buffers->inflatablePressures.map();

	buffers->triangles.map();
	buffers->triangleNormals.map();
	buffers->uvs.map();
}

void UnmapBuffers(SimBuffers* buffers)
{
	// particles
	buffers->positions.unmap();
	buffers->restPositions.unmap();
	buffers->velocities.unmap();
	buffers->phases.unmap();
	buffers->densities.unmap();
	buffers->anisotropy1.unmap();
	buffers->anisotropy2.unmap();
	buffers->anisotropy3.unmap();
	buffers->normals.unmap();
	buffers->diffusePositions.unmap();
	buffers->diffuseVelocities.unmap();
	buffers->diffuseCount.unmap();
	buffers->smoothPositions.unmap();
	buffers->activeIndices.unmap();

	// convexes
	buffers->shapeGeometry.unmap();
	buffers->shapePositions.unmap();
	buffers->shapeRotations.unmap();
	buffers->shapePrevPositions.unmap();
	buffers->shapePrevRotations.unmap();
	buffers->shapeFlags.unmap();

	// rigids
	buffers->rigidOffsets.unmap();
	buffers->rigidIndices.unmap();
	buffers->rigidMeshSize.unmap();
	buffers->rigidCoefficients.unmap();
	buffers->rigidPlasticThresholds.unmap();
	buffers->rigidPlasticCreeps.unmap();
	buffers->rigidRotations.unmap();
	buffers->rigidTranslations.unmap();
	buffers->rigidLocalPositions.unmap();
	buffers->rigidLocalNormals.unmap();

	// springs
	buffers->springIndices.unmap();
	buffers->springLengths.unmap();
	buffers->springStiffness.unmap();

	// inflatables
	buffers->inflatableTriOffsets.unmap();
	buffers->inflatableTriCounts.unmap();
	buffers->inflatableVolumes.unmap();
	buffers->inflatableCoefficients.unmap();
	buffers->inflatablePressures.unmap();

	// triangles
	buffers->triangles.unmap();
	buffers->triangleNormals.unmap();
	buffers->uvs.unmap();

}

SimBuffers* AllocBuffers(NvFlexLibrary* lib)
{
	return new SimBuffers(lib);
}

void DestroyBuffers(SimBuffers* buffers)
{
	// particles
	buffers->positions.destroy();
	buffers->restPositions.destroy();
	buffers->velocities.destroy();
	buffers->phases.destroy();
	buffers->densities.destroy();
	buffers->anisotropy1.destroy();
	buffers->anisotropy2.destroy();
	buffers->anisotropy3.destroy();
	buffers->normals.destroy();
	buffers->diffusePositions.destroy();
	buffers->diffuseVelocities.destroy();
	buffers->diffuseCount.destroy();
	buffers->smoothPositions.destroy();
	buffers->activeIndices.destroy();

	// convexes
	buffers->shapeGeometry.destroy();
	buffers->shapePositions.destroy();
	buffers->shapeRotations.destroy();
	buffers->shapePrevPositions.destroy();
	buffers->shapePrevRotations.destroy();
	buffers->shapeFlags.destroy();

	// rigids
	buffers->rigidOffsets.destroy();
	buffers->rigidIndices.destroy();
	buffers->rigidMeshSize.destroy();
	buffers->rigidCoefficients.destroy();
	buffers->rigidPlasticThresholds.destroy();
	buffers->rigidPlasticCreeps.destroy();
	buffers->rigidRotations.destroy();
	buffers->rigidTranslations.destroy();
	buffers->rigidLocalPositions.destroy();
	buffers->rigidLocalNormals.destroy();

	// springs
	buffers->springIndices.destroy();
	buffers->springLengths.destroy();
	buffers->springStiffness.destroy();

	// inflatables
	buffers->inflatableTriOffsets.destroy();
	buffers->inflatableTriCounts.destroy();
	buffers->inflatableVolumes.destroy();
	buffers->inflatableCoefficients.destroy();
	buffers->inflatablePressures.destroy();

	// triangles
	buffers->triangles.destroy();
	buffers->triangleNormals.destroy();
	buffers->uvs.destroy();

	delete buffers;
}

void GetParticleBounds(Vec3& lower, Vec3& upper)
{
	lower = Vec3(FLT_MAX);
	upper = Vec3(-FLT_MAX);

	for (int i = 0; i < g_buffers->positions.size(); ++i)
	{
		lower = Min(Vec3(g_buffers->positions[i]), lower);
		upper = Max(Vec3(g_buffers->positions[i]), upper);
	}
}

inline int GridIndex(int x, int y, int dx) { return y * dx + x; }

void CreateSpring(int i, int j, float stiffness, float give = 0.0f)
{
	g_buffers->springIndices.push_back(i);
	g_buffers->springIndices.push_back(j);
	g_buffers->springLengths.push_back((1.0f + give)*Length(Vec3(g_buffers->positions[i]) - Vec3(g_buffers->positions[j])));
	g_buffers->springStiffness.push_back(stiffness);
}

void CreateParticleGrid(Vec3 lower, int dimx, int dimy, int dimz, float radius, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, float jitter = 0.005f)
{
	if (rigid && g_buffers->rigidIndices.empty())
		g_buffers->rigidOffsets.push_back(0);

	for (int x = 0; x < dimx; ++x)
	{
		for (int y = 0; y < dimy; ++y)
		{
			for (int z = 0; z < dimz; ++z)
			{
				if (rigid)
					g_buffers->rigidIndices.push_back(int(g_buffers->positions.size()));

				Vec3 position = lower + Vec3(float(x), float(y), float(z))*radius + RandomUnitVector()*jitter;

				g_buffers->positions.push_back(Vec4(position.x, position.y, position.z, invMass));
				g_buffers->velocities.push_back(velocity);
				g_buffers->phases.push_back(phase);

				initPos.push_back(daedalus::Vec4f(position.x, position.y, position.z, 1.0f));
				initCol.push_back(daedalus::Vec4f(0.0f, 0.5f, 1.0f, 1.0f));
			}
		}
	}

	if (rigid)
	{
		g_buffers->rigidCoefficients.push_back(rigidStiffness);
		g_buffers->rigidOffsets.push_back(int(g_buffers->rigidIndices.size()));
	}
}

void CreateSpringGrid(Vec3 lower, int dx, int dy, int dz, float radius, int phase, float stretchStiffness, float bendStiffness, float shearStiffness, Vec3 velocity, float invMass)
{
	int baseIndex = int(g_buffers->positions.size());

	for (int z = 0; z < dz; ++z)
	{
		for (int y = 0; y < dy; ++y)
		{
			for (int x = 0; x < dx; ++x)
			{
				Vec3 position = lower + radius * Vec3(float(x), float(z), float(y));

				g_buffers->positions.push_back(Vec4(position.x, position.y, position.z, invMass));
				g_buffers->velocities.push_back(velocity);
				g_buffers->phases.push_back(phase);

				initPos.push_back(daedalus::Vec4f(position.x, position.y, position.z, 1.0f));
				initCol.push_back(daedalus::Vec4f(0.3f, 1.0f, 0.3f, 1.0f));

				if (x > 0 && y > 0)
				{
					g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y - 1, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x, y - 1, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x, y, dx));

					g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y - 1, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x, y, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y, dx));

					g_buffers->triangleNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
					g_buffers->triangleNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
				}
			}
		}
	}

	// horizontal
	for (int y = 0; y < dy; ++y)
	{
		for (int x = 0; x < dx; ++x)
		{
			int index0 = y * dx + x;

			if (x > 0)
			{
				int index1 = y * dx + x - 1;
				CreateSpring(baseIndex + index0, baseIndex + index1, stretchStiffness);
			}

			if (x > 1)
			{
				int index2 = y * dx + x - 2;
				CreateSpring(baseIndex + index0, baseIndex + index2, bendStiffness);
			}

			if (y > 0 && x < dx - 1)
			{
				int indexDiag = (y - 1)*dx + x + 1;
				CreateSpring(baseIndex + index0, baseIndex + indexDiag, shearStiffness);
			}

			if (y > 0 && x > 0)
			{
				int indexDiag = (y - 1)*dx + x - 1;
				CreateSpring(baseIndex + index0, baseIndex + indexDiag, shearStiffness);
			}
		}
	}

	// vertical
	for (int x = 0; x < dx; ++x)
	{
		for (int y = 0; y < dy; ++y)
		{
			int index0 = y * dx + x;

			if (y > 0)
			{
				int index1 = (y - 1)*dx + x;
				CreateSpring(baseIndex + index0, baseIndex + index1, stretchStiffness);
			}

			if (y > 1)
			{
				int index2 = (y - 2)*dx + x;
				CreateSpring(baseIndex + index0, baseIndex + index2, bendStiffness);
			}
		}
	}
}

#include "scenes.h"

class DamBreak : public Scene
{
public:

	DamBreak(const char* name, float radius) : Scene(name), mRadius(radius) {}

	virtual void Initialize()
	{
		const float radius = mRadius;
		const float restDistance = mRadius * 0.65f;

		int dx = int(ceilf(1.0f / restDistance));
		int dy = int(ceilf(2.0f / restDistance));
		int dz = int(ceilf(1.0f / restDistance));

		CreateParticleGrid(Vec3(0.0f, restDistance, 0.0f), dx, dy, dz, restDistance, Vec3(0.0f), 1.0f, false, 0.0f, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid), restDistance*0.01f);

		g_sceneLower = Vec3(0.0f, 0.0f, -0.5f);
		g_sceneUpper = Vec3(3.0f, 0.0f, -0.5f);

		g_numSubsteps = 2;

		g_params.radius = radius;
		g_params.fluidRestDistance = restDistance;
		g_params.dynamicFriction = 0.f;
		g_params.restitution = 0.001f;

		g_params.numIterations = 3;
		g_params.relaxationFactor = 1.0f;

		g_params.smoothing = 0.4f;

		g_params.viscosity = 0.001f;
		g_params.cohesion = 0.1f;
		g_params.vorticityConfinement = 80.0f;
		g_params.surfaceTension = 0.0f;

		g_params.numPlanes = 5;

		// limit velocity to CFL condition
		g_params.maxSpeed = 0.5f*radius*g_numSubsteps / g_dt;

		g_maxDiffuseParticles = 0;
	}

	float mRadius;
};

class FlagCloth : public Scene
{
public:

	FlagCloth(const char* name) : Scene(name) {}

	void Initialize()
	{
		int dimx = 64;
		int dimz = 32;
		float radius = 0.05f;

		float stretchStiffness = 0.9f;
		float bendStiffness = 1.0f;
		float shearStiffness = 0.9f;
		int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);

		CreateSpringGrid(Vec3(0.0f, 0.0f, -3.0f), dimx, dimz, 1, radius, phase, stretchStiffness, bendStiffness, shearStiffness, 0.0f, 1.0f);

		const int c1 = 0;
		const int c2 = dimx * (dimz - 1);

		g_buffers->positions[c1].w = 0.0f;
		g_buffers->positions[c2].w = 0.0f;

		// add tethers
		for (int i = 0; i < int(g_buffers->positions.size()); ++i)
		{
			// hack to rotate cloth
			std::swap(g_buffers->positions[i].y, g_buffers->positions[i].z);
			g_buffers->positions[i].y *= -1.0f;

			g_buffers->velocities[i] = RandomUnitVector()*0.1f;

			float minSqrDist = FLT_MAX;

			if (i != c1 && i != c2)
			{
				float stiffness = -0.8f;
				float give = 0.1f;

				float sqrDist = LengthSq(Vec3(g_buffers->positions[c1]) - Vec3(g_buffers->positions[c2]));

				if (sqrDist < minSqrDist)
				{
					CreateSpring(c1, i, stiffness, give);
					CreateSpring(c2, i, stiffness, give);

					minSqrDist = sqrDist;
				}
			}
		}

		g_params.radius = radius * 1.0f;
		g_params.dynamicFriction = 0.25f;
		g_params.dissipation = 0.0f;
		g_params.numIterations = 4;
		g_params.drag = 0.06f;
		g_params.relaxationFactor = 1.0f;

		g_numSubsteps = 2;

		// draw options
		g_windFrequency *= 2.0f;
		g_windStrength = 10.0f;
	}

	void Update()
	{
		const Vec3 kWindDir = Vec3(3.0f, 15.0f, 0.0f);
		const float kNoise = fabsf(Perlin1D(g_windTime*0.05f, 2, 0.25f));
		Vec3 wind = g_windStrength * kWindDir*Vec3(kNoise, kNoise*0.1f, -kNoise * 0.1f);

		g_params.wind[0] = wind.x;
		g_params.wind[1] = wind.y;
		g_params.wind[2] = wind.z;
	}
};

class Scene;
std::vector<Scene*> g_scenes;

void UpdateWind()
{
	g_windTime += g_dt;

	const Vec3 kWindDir = Vec3(3.0f, 15.0f, 0.0f);
	const float kNoise = Perlin1D(g_windTime*g_windFrequency, 10, 0.25f);
	Vec3 wind = g_windStrength * kWindDir*Vec3(kNoise, fabsf(kNoise), 0.0f);

	g_params.wind[0] = wind.x;
	g_params.wind[1] = wind.y;
	g_params.wind[2] = wind.z;

	if (g_wavePool)
	{
		g_waveTime += g_dt;

		g_params.planes[2][3] = g_wavePlane + (sinf(float(g_waveTime)*g_waveFrequency - kPi * 0.5f)*0.5f + 0.5f)*g_waveAmplitude;
	}
}

void UpdateScene()
{
	// give scene a chance to make changes to particle buffers
	g_scenes[0]->Update();
}

void ErrorCallback(NvFlexErrorSeverity severity, const char* msg, const char* file, int line)
{
	printf("Flex: %s - %s:%d\n", msg, file, line);
	g_Error = (severity == eNvFlexLogError);
	//assert(0); asserts are bad for TeamCity
}

class App : public GLUTApplication
{
public:
	App();
	~App();

public:
	virtual void PrintCommandLineHelp();

	virtual void Initialize();
	virtual void Update();
	virtual void Render();
	virtual void Terminate();

	virtual std::string GetName();

	void InitFlexScene()
	{
		g_scenes.clear();

#ifdef flex_dambreak
		g_scenes.push_back(new DamBreak("DamBreak  5cm", 0.05f));
#endif
#ifdef flex_flagcloth
		g_scenes.push_back(new FlagCloth("Flag Cloth"));
#endif
		NvFlexInitDesc desc;
		desc.deviceIndex = g_device;
		desc.enableExtensions = g_extensions;
		desc.renderDevice = 0;
		desc.renderContext = 0;
		desc.computeContext = 0;
		desc.computeType = eNvFlexCUDA;

		// Init Flex library, note that no CUDA methods should be called before this 
		// point to ensure we get the device context we want
		g_flexLib = NvFlexInit(NV_FLEX_VERSION, ErrorCallback, &desc);

		if (g_Error || g_flexLib == NULL)
		{
			printf("Could not initialize Flex, exiting.\n");
			exit(-1);
		}

		// store device name
		strcpy(g_deviceName, NvFlexGetDeviceName(g_flexLib));
		printf("Flex Compute Device: %s\n\n", g_deviceName);

		//Init(g_scene);
		{
			RandInit();

			if (g_solver)
			{
				// destroy + reset
			}

			// alloc buffers
			g_buffers = AllocBuffers(g_flexLib);

			// map during initialization
			MapBuffers(g_buffers);

			g_buffers->positions.resize(0);
			g_buffers->velocities.resize(0);
			g_buffers->phases.resize(0);

			g_buffers->rigidOffsets.resize(0);
			g_buffers->rigidIndices.resize(0);
			g_buffers->rigidMeshSize.resize(0);
			g_buffers->rigidRotations.resize(0);
			g_buffers->rigidTranslations.resize(0);
			g_buffers->rigidCoefficients.resize(0);
			g_buffers->rigidPlasticThresholds.resize(0);
			g_buffers->rigidPlasticCreeps.resize(0);
			g_buffers->rigidLocalPositions.resize(0);
			g_buffers->rigidLocalNormals.resize(0);

			g_buffers->springIndices.resize(0);
			g_buffers->springLengths.resize(0);
			g_buffers->springStiffness.resize(0);
			g_buffers->triangles.resize(0);
			g_buffers->triangleNormals.resize(0);
			g_buffers->uvs.resize(0);

			//g_meshSkinIndices.resize(0);
			//g_meshSkinWeights.resize(0);

			//g_emitters.resize(1);
			//g_emitters[0].mEnabled = false;
			//g_emitters[0].mSpeed = 1.0f;
			//g_emitters[0].mLeftOver = 0.0f;
			//g_emitters[0].mWidth = 8;

			g_buffers->shapeGeometry.resize(0);
			g_buffers->shapePositions.resize(0);
			g_buffers->shapeRotations.resize(0);
			g_buffers->shapePrevPositions.resize(0);
			g_buffers->shapePrevRotations.resize(0);
			g_buffers->shapeFlags.resize(0);

			//g_ropes.resize(0);

			// remove collision shapes
			//delete g_mesh; g_mesh = NULL;

			g_frame = 0;
			g_pause = false;

			g_dt = 1.0f / 60.0f;
			g_waveTime = 0.0f;
			g_windTime = 0.0f;
			g_windStrength = 1.0f;
			
			//g_blur = 1.0f;
			//g_fluidColor = Vec4(0.1f, 0.4f, 0.8f, 1.0f);
			//g_meshColor = Vec3(0.9f, 0.9f, 0.9f);
			//g_drawEllipsoids = false;
			//g_drawPoints = true;
			//g_drawCloth = true;
			//g_expandCloth = 0.0f;
			//
			//g_drawOpaque = false;
			//g_drawSprings = false;
			//g_drawDiffuse = false;
			//g_drawMesh = true;
			//g_drawRopes = true;
			//g_drawDensity = false;
			//g_ior = 1.0f;
			//g_lightDistance = 2.0f;
			//g_fogDistance = 0.005f;
			//
			//g_camSpeed = 0.075f;
			//g_camNear = 0.01f;
			//g_camFar = 1000.0f;

			g_pointScale = 1.0f;
			g_ropeScale = 1.0f;
			g_drawPlaneBias = 0.0f;

			// sim params
			g_params.gravity[0] = 0.0f;
			g_params.gravity[1] = -9.8f;
			g_params.gravity[2] = 0.0f;

			g_params.wind[0] = 0.0f;
			g_params.wind[1] = 0.0f;
			g_params.wind[2] = 0.0f;

			g_params.radius = 0.15f;
			g_params.viscosity = 0.0f;
			g_params.dynamicFriction = 0.0f;
			g_params.staticFriction = 0.0f;
			g_params.particleFriction = 0.0f; // scale friction between particles by default
			g_params.freeSurfaceDrag = 0.0f;
			g_params.drag = 0.0f;
			g_params.lift = 0.0f;
			g_params.numIterations = 3;
			g_params.fluidRestDistance = 0.0f;
			g_params.solidRestDistance = 0.0f;

			g_params.anisotropyScale = 1.0f;
			g_params.anisotropyMin = 0.1f;
			g_params.anisotropyMax = 2.0f;
			g_params.smoothing = 1.0f;

			g_params.dissipation = 0.0f;
			g_params.damping = 0.0f;
			g_params.particleCollisionMargin = 0.0f;
			g_params.shapeCollisionMargin = 0.0f;
			g_params.collisionDistance = 0.0f;
			g_params.sleepThreshold = 0.0f;
			g_params.shockPropagation = 0.0f;
			g_params.restitution = 0.0f;

			g_params.maxSpeed = FLT_MAX;
			g_params.maxAcceleration = 100.0f;	// approximately 10x gravity

			g_params.relaxationMode = eNvFlexRelaxationLocal;
			g_params.relaxationFactor = 1.0f;
			g_params.solidPressure = 1.0f;
			g_params.adhesion = 0.0f;
			g_params.cohesion = 0.025f;
			g_params.surfaceTension = 0.0f;
			g_params.vorticityConfinement = 0.0f;
			g_params.buoyancy = 1.0f;
			g_params.diffuseThreshold = 100.0f;
			g_params.diffuseBuoyancy = 1.0f;
			g_params.diffuseDrag = 0.8f;
			g_params.diffuseBallistic = 16;
			g_params.diffuseLifetime = 2.0f;

			g_numSubsteps = 2;

			// planes created after particles
			g_params.numPlanes = 1;

			//g_diffuseScale = 0.5f;
			//g_diffuseColor = 1.0f;
			//g_diffuseMotionScale = 1.0f;
			//g_diffuseShadow = false;
			//g_diffuseInscatter = 0.8f;
			//g_diffuseOutscatter = 0.53f;

			// reset phase 0 particle color to blue
			//g_colors[0] = Colour(0.0f, 0.5f, 1.0f);

			//g_numSolidParticles = 0;

			g_waveFrequency = 1.5f;
			g_waveAmplitude = 1.5f;
			g_waveFloorTilt = 0.0f;
			g_emit = false;
			g_warmup = false;

			g_mouseParticle = -1;

			g_maxDiffuseParticles = 0;	// number of diffuse particles
			g_maxNeighborsPerParticle = 96;
			g_numExtraParticles = 0;	// number of particles allocated but not made active	
			g_maxContactsPerParticle = 6;

			g_sceneLower = FLT_MAX;
			g_sceneUpper = -FLT_MAX;

			// initialize solver desc
			NvFlexSetSolverDescDefaults(&g_solverDesc);

			g_scenes[0]->Initialize();

			uint32_t numParticles = g_buffers->positions.size();
			uint32_t maxParticles = numParticles + g_numExtraParticles * g_numExtraMultiplier;

			if (g_params.solidRestDistance == 0.0f)
				g_params.solidRestDistance = g_params.radius;

			// if fluid present then we assume solid particles have the same radius
			if (g_params.fluidRestDistance > 0.0f)
				g_params.solidRestDistance = g_params.fluidRestDistance;

			// set collision distance automatically based on rest distance if not alraedy set
			if (g_params.collisionDistance == 0.0f)
				g_params.collisionDistance = Max(g_params.solidRestDistance, g_params.fluidRestDistance)*0.5f;

			// default particle friction to 10% of shape friction
			if (g_params.particleFriction == 0.0f)
				g_params.particleFriction = g_params.dynamicFriction*0.1f;

			// add a margin for detecting contacts between particles and shapes
			if (g_params.shapeCollisionMargin == 0.0f)
				g_params.shapeCollisionMargin = g_params.collisionDistance*0.5f;

			// calculate particle bounds
			Vec3 particleLower, particleUpper;
			GetParticleBounds(particleLower, particleUpper);

			// update bounds
			g_sceneLower = Min(g_sceneLower, particleLower);
			g_sceneUpper = Max(g_sceneUpper, particleUpper);

			g_sceneLower -= g_params.collisionDistance;
			g_sceneUpper += g_params.collisionDistance;

			// update collision planes to match flexs
			Vec3 up = Normalize(Vec3(-g_waveFloorTilt, 1.0f, 0.0f));

			(Vec4&)g_params.planes[0] = Vec4(up.x, up.y, up.z, 0.0f);
			(Vec4&)g_params.planes[1] = Vec4(0.0f, 0.0f, 1.0f, -g_sceneLower.z);
			(Vec4&)g_params.planes[2] = Vec4(1.0f, 0.0f, 0.0f, -g_sceneLower.x);
			(Vec4&)g_params.planes[3] = Vec4(-1.0f, 0.0f, 0.0f, g_sceneUpper.x);
			(Vec4&)g_params.planes[4] = Vec4(0.0f, 0.0f, -1.0f, g_sceneUpper.z);
			(Vec4&)g_params.planes[5] = Vec4(0.0f, -1.0f, 0.0f, g_sceneUpper.y);

			g_wavePlane = g_params.planes[2][3];

			g_buffers->diffusePositions.resize(g_maxDiffuseParticles);
			g_buffers->diffuseVelocities.resize(g_maxDiffuseParticles);
			g_buffers->diffuseCount.resize(1, 0);

			// for fluid rendering these are the Laplacian smoothed positions
			g_buffers->smoothPositions.resize(maxParticles);

			g_buffers->normals.resize(0);
			g_buffers->normals.resize(maxParticles);

			// initialize normals (just for rendering before simulation starts)
			int numTris = g_buffers->triangles.size() / 3;
			for (int i = 0; i < numTris; ++i)
			{
				Vec3 v0 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 0]]);
				Vec3 v1 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 1]]);
				Vec3 v2 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 2]]);

				Vec3 n = Cross(v1 - v0, v2 - v0);

				g_buffers->normals[g_buffers->triangles[i * 3 + 0]] += Vec4(n, 0.0f);
				g_buffers->normals[g_buffers->triangles[i * 3 + 1]] += Vec4(n, 0.0f);
				g_buffers->normals[g_buffers->triangles[i * 3 + 2]] += Vec4(n, 0.0f);
			}

			for (int i = 0; i < int(maxParticles); ++i)
				g_buffers->normals[i] = Vec4(SafeNormalize(Vec3(g_buffers->normals[i]), Vec3(0.0f, 1.0f, 0.0f)), 0.0f);

			g_solverDesc.maxParticles = maxParticles;
			g_solverDesc.maxDiffuseParticles = g_maxDiffuseParticles;
			g_solverDesc.maxNeighborsPerParticle = g_maxNeighborsPerParticle;
			g_solverDesc.maxContactsPerParticle = g_maxContactsPerParticle;

			// main create method for the Flex solver
			g_solver = NvFlexCreateSolver(g_flexLib, &g_solverDesc);

			// give scene a chance to do some post solver initialization
			g_scenes[0]->PostInitialize();

			// create active indices (just a contiguous block for the demo)
			g_buffers->activeIndices.resize(g_buffers->positions.size());
			for (int i = 0; i < g_buffers->activeIndices.size(); ++i)
				g_buffers->activeIndices[i] = i;

			// resize particle buffers to fit
			g_buffers->positions.resize(maxParticles);
			g_buffers->velocities.resize(maxParticles);
			g_buffers->phases.resize(maxParticles);

			g_buffers->densities.resize(maxParticles);
			g_buffers->anisotropy1.resize(maxParticles);
			g_buffers->anisotropy2.resize(maxParticles);
			g_buffers->anisotropy3.resize(maxParticles);

			// save rest positions
			g_buffers->restPositions.resize(g_buffers->positions.size());
			for (int i = 0; i < g_buffers->positions.size(); ++i)
				g_buffers->restPositions[i] = g_buffers->positions[i];

			// unmap so we can start transferring data to GPU
			UnmapBuffers(g_buffers);

			//-----------------------------
			// Send data to Flex

			NvFlexCopyDesc copyDesc;
			copyDesc.dstOffset = 0;
			copyDesc.srcOffset = 0;
			copyDesc.elementCount = numParticles;

			NvFlexSetParams(g_solver, &g_params);
			NvFlexSetParticles(g_solver, g_buffers->positions.buffer, &copyDesc);
			NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, &copyDesc);
			NvFlexSetNormals(g_solver, g_buffers->normals.buffer, &copyDesc);
			NvFlexSetPhases(g_solver, g_buffers->phases.buffer, &copyDesc);
			NvFlexSetRestParticles(g_solver, g_buffers->restPositions.buffer, &copyDesc);

			NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, &copyDesc);
			NvFlexSetActiveCount(g_solver, numParticles);

			// springs
			if (g_buffers->springIndices.size())
			{
				assert((g_buffers->springIndices.size() & 1) == 0);
				assert((g_buffers->springIndices.size() / 2) == g_buffers->springLengths.size());

				NvFlexSetSprings(g_solver, g_buffers->springIndices.buffer, g_buffers->springLengths.buffer, g_buffers->springStiffness.buffer, g_buffers->springLengths.size());
			}

			// dynamic triangles
			if (g_buffers->triangles.size())
			{
				NvFlexSetDynamicTriangles(g_solver, g_buffers->triangles.buffer, g_buffers->triangleNormals.buffer, g_buffers->triangles.size() / 3);
			}
		}
	}

	void UpdateFlexFrame()
	{
		// Scene Update
		{
			MapBuffers(g_buffers);

			if (!g_pause || g_step)
			{
				//UpdateEmitters();
				//UpdateMouse();
				UpdateWind();
				UpdateScene();

				{
					glBindBuffer(GL_ARRAY_BUFFER, vboPos);
					void* hostPtr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
					memcpy(hostPtr, &g_buffers->positions[0], g_buffers->positions.size() * 4 * sizeof(float));
					glUnmapBuffer(GL_ARRAY_BUFFER);
					glBindBuffer(GL_ARRAY_BUFFER, 0);
				}
			}

			UnmapBuffers(g_buffers);
		}

		// Flex Update
		{
			// send any particle updates to the solver
			NvFlexSetParticles(g_solver, g_buffers->positions.buffer, NULL);
			NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, NULL);
			NvFlexSetPhases(g_solver, g_buffers->phases.buffer, NULL);
			NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, NULL);

			NvFlexSetActiveCount(g_solver, g_buffers->activeIndices.size());

			// allow scene to update constraints etc
			//SyncScene();

			if (!g_pause || g_step)
			{
				// tick solver
				NvFlexSetParams(g_solver, &g_params);
				NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);

				g_frame++;
				g_step = false;
			}

			NvFlexGetParticles(g_solver, g_buffers->positions.buffer, NULL);
			NvFlexGetVelocities(g_solver, g_buffers->velocities.buffer, NULL);
			NvFlexGetNormals(g_solver, g_buffers->normals.buffer, NULL);
		}
	}

protected:
	daedalus::Camera mainCamera;

	GLuint vboPos;
	GLuint vboCol;

	GLMesh meshWorldAABB;
	
	GLProgram renderProgram_BBox;
	GLProgram renderProgram_Particles;
};