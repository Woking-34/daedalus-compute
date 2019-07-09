#ifndef MESH_H
#define MESH_H

#include "math/mathbase.h"
#include "math/core/vec.h"
#include "math/core/mat.h"
#include "math/util/aabb.h"

#include <string>
#include <vector>

enum MeshPrimitiveType
{
	MESHPRIM_TYPE_NOTSET,
	MESHPRIM_TYPE_POINTS,
	MESHPRIM_TYPE_LINES,
	MESHPRIM_TYPE_TRIS,
	MESHPRIM_TYPE_TRI_STRIP,
	MESHPRIM_TYPE_QUADS,
	MESHPRIM_TYPE_QUAD_STRIP,
	MESHPRIM_TYPE_POLYS
};

class MeshFile
{
public:
	MeshFile() : numAttribs(8)
	{
		reset();
	}
	~MeshFile() {}

	void setName(const std::string& fileName, const std::string& fileNameMtlBase);
	void loadFromFile(bool forceFileLoad = false);

	void createFSQPos();
	void createFSQPosTex();

	void createWorldAxis(float scale = 1.0f);

	void createGridTris(float w, float l, unsigned int wSegs, unsigned int lSegs);
	void createGridWire(float w, float l, unsigned int wSegs, unsigned int lSegs);
	
	void createBoxTris(float w, float h, float l);
	void createBoxWire(float w, float h, float l);

	void createSphereTris(float rad, unsigned int hSegs, unsigned int vSegs);
	void createSphereWire(float rad, unsigned int segs);

	void createTorusTris(float rad0, float rad1, unsigned int hSegs, unsigned int vSegs);
	void createTorusWire(float rad0, float rad1, unsigned int hSegs, unsigned int vSegs);

	void createConeTris(float height, float angle, unsigned int segs);
	void createConeWire(float height, float angle, unsigned int segs);

	std::vector<daedalus::Vec4f> position0Vec;
	std::vector<daedalus::Vec4f> position1Vec;
	std::vector<daedalus::Vec4f> normals0Vec;
	std::vector<daedalus::Vec4f> normals1Vec;
	std::vector<daedalus::Vec4f> colors0Vec;
	std::vector<daedalus::Vec4f> colors1Vec;
	std::vector<daedalus::Vec2f> uvcoord0Vec;
	std::vector<daedalus::Vec2f> uvcoord1Vec;

	std::vector< std::vector<unsigned int> > boneindicesVec;
	std::vector< std::vector<float> > boneweightsVec;

	int numAttribs;
	std::string attribName[8];
	bool hasAttribute[8];

	MeshPrimitiveType primType;
	unsigned int numVertsPrim;

	unsigned int numPrimitives;
	unsigned int numVertices;

	std::vector< std::vector<unsigned int> > indicesVec;
	std::vector< unsigned int > indicesVecFlat;
	std::vector< unsigned int > attributeVec;
	std::vector< std::string > usemtlVec;

	// util functions
	daedalus::AABB4f getAABB() const;

	void generateNormals();
	void copyNormalsToColors();

	void setPositionVec(const daedalus::Vec4f* x, unsigned int numElements);
	void setNormalVec(const daedalus::Vec4f* x, unsigned int numElements);
	void setColorVec(const daedalus::Vec4f* x, unsigned int numElements);
	void setColorVec(float r, float g, float b);

	void add(const MeshFile& meshdata, const daedalus::Mat44f& mat);

	// cornell box helper
	void addParallelogram(const daedalus::Vec4f& anchor, const daedalus::Vec4f& offset1, const daedalus::Vec4f& offset2, const daedalus::Vec4f& c);

	void triangulate();
	void tesselate();

	void scale(float s);
	void scale(const daedalus::Vec4f& scaleVec);

	void translate(float x, float y, float z);
	void translate(const daedalus::Vec4f& translateVec);

	void scaleToUnitCube();
	void placeToXZPlane();

	void flattenIndices();

	void saveRaw() const;
	bool loadRaw();

	bool isFoundSource;
	bool isLoadedFromCached;

	std::string coreName;
	std::string extName;

	std::string fileName;
	std::string fileNameSub;
	std::string fileNameFull;

	void printInfo();

protected:
	void reset();

	void fillAttributeVec();

	bool loadFromPLY(const std::string& fileName);
	bool loadFromOBJ(const std::string& fileName, const std::string& fileNameMtlBase);

	struct VertexDescObj
	{
		VertexDescObj()
		{
			positionId = uvcoordId = normalId = 0;
		};

		int positionId;
		int uvcoordId;
		int normalId;

		int materialId;
	};

	struct MaterialDescObj
	{
		MaterialDescObj()
		{
			ka = kd = ks = daedalus::Vec4f(0.0f, 0.0f, 0.0f, 1.0f);
		};

		std::string name;

		daedalus::Vec4f ka, kd, ks;

		std::string diffMap;
		std::string specMap;
		std::string bumpMap;
	};

	std::vector<MaterialDescObj> materialVec;
};

#endif