#include "meshfile.h"

#include "system/platform.h"
#include "system/filesystem.h"
#include "system/log.h"

using namespace daedalus;

void MeshFile::setName(const std::string& fileName, const std::string& fileNameSub)
{
	this->coreName = fileName.substr(0, fileName.size()-4);
	this->extName = fileName.substr(fileName.length()-3,3);

	this->fileName = fileName;
	this->fileNameSub = fileNameSub + fileName;
	
	fileNameFull = this->fileNameSub;
	isFoundSource = findFullPath(fileNameFull);
}

void MeshFile::printInfo()
{
	LOG_OK( LogLine() << "MESHFILE" << " - " << coreName << "." << extName );
	LOG_OK( LogLine() << "MESHFILE VERTS" << " - " << numVertices );
	LOG_OK( LogLine() << "MESHFILE PRIMS" << " - " << numPrimitives );
	LOG("");
}

void MeshFile::loadFromFile(bool forceFileLoad)
{
	LOG_BOOL( isFoundSource, LogLine() << "MESHFILE FILEOPEN" << " - " << fileName );

	std::string fileExtension = fileName.substr(fileName.length()-3,3);

	if(fileExtension == "ply" || fileExtension == "PLY")
	{
		loadFromPLY(fileNameFull);
	}
	else if(fileExtension == "obj" || fileExtension == "OBJ")
	{
		loadFromOBJ(fileNameFull, "");
	}
	else
	{
		LOG_ERR( LogLine() << "MESHFILE FILETYPE_UNSUPPORTED" << " - " << fileName );
	}

	fillAttributeVec();
	printInfo();
}

bool MeshFile::loadFromPLY(const std::string& fileName)
{
	std::ifstream inFile(fileName.c_str(), std::ios::in | std::ios::binary);
	if( inFile.is_open() )
	{
		std::string fileLineStr;

		// check header opening "ply tag"
		getline(inFile, fileLineStr);
		if( fileLineStr.find("ply") == std::string::npos )
		{
			std::cout << "[ERR] MESHFILE PLYFILE_FORMATERROR" << " - " << fileName << std::endl;
			return false;
		}

		bool isASCII = true;
		bool needEndianSwap = false;
		bool isStripped = false;

		// check header format type
		getline(inFile, fileLineStr);
		if( fileLineStr.find("ascii") != std::string::npos )
		{
			isASCII = true;
		}
		else if( fileLineStr.find("binary") != std::string::npos )
		{
			isASCII = false;
		}

		if( fileLineStr.find("big_endian") != std::string::npos )
		{
			needEndianSwap = true;
		}

		while( getline(inFile, fileLineStr) )
		{
			// not a blank line -> skip
			if(fileLineStr.size() == 0)
			{
				continue;
			}

			std::string fileLineCommand;

			std::stringstream fileLineSS;
			fileLineSS << fileLineStr;
			fileLineSS >> fileLineCommand;

			if(fileLineCommand == "element")
			{
				fileLineSS >> fileLineCommand;

				if(fileLineCommand == "vertex")
				{
					fileLineSS >> numVertices;
				}

				if(fileLineCommand == "face")
				{
					fileLineSS >> numPrimitives;
				}

				if(fileLineCommand == "tristrips")
				{
					isStripped = true;
					fileLineSS >> numPrimitives;
				}
			}

			if(fileLineCommand == "end_header")
			{
				break;
			}
		}

		if(isASCII)
		{
			for(unsigned int i = 0; i < numVertices; ++i)
			{
				getline(inFile, fileLineStr);

				std::stringstream fileLineSS;
				fileLineSS << fileLineStr;

				Vec4f vert;

				fileLineSS >> vert.x;
				fileLineSS >> vert.y;
				fileLineSS >> vert.z;
				vert.w = 1.0f;

				position0Vec.push_back(vert);
			}

			for(unsigned int i = 0; i < numPrimitives; ++i)
			{
				getline(inFile, fileLineStr);

				std::stringstream fileLineSS;
				fileLineSS << fileLineStr;

				unsigned int numVertsInPrim = 0;
				fileLineSS >> numVertsInPrim;

				indicesVec.push_back( std::vector<unsigned int>() );

				for(unsigned int v = 0; v < numVertsInPrim; ++v)
				{
					unsigned int vertId = 0;
					fileLineSS >> vertId;

					indicesVec[i].push_back(vertId);
				}
			}

			primType = MESHPRIM_TYPE_TRIS;
			numVertsPrim = 3;
		}
		else
		{
			for(unsigned int i = 0; i < numVertices; ++i)
			{
				Vec4f vert;

				inFile.read(reinterpret_cast<char*>(&(vert.x)), sizeof(float));
				inFile.read(reinterpret_cast<char*>(&(vert.y)), sizeof(float));
				inFile.read(reinterpret_cast<char*>(&(vert.z)), sizeof(float));
				vert.w = 1.0f;

				if(needEndianSwap)
				{
					endianswap( &(vert.x) );
					endianswap( &(vert.y) );
					endianswap( &(vert.z) );
				}

				position0Vec.push_back(vert);
			}

			for(unsigned int i = 0; i < numPrimitives; ++i)
			{
				unsigned char numVertsInPrim = 0;
				inFile.read(reinterpret_cast<char*>(&numVertsInPrim), sizeof(unsigned char));

				if(needEndianSwap)
				{
				//	endswap( &numVertsInPrim );
				}

				indicesVec.push_back( std::vector<unsigned int>() );

				for(unsigned char v = 0; v < numVertsInPrim; ++v)
				{
					unsigned int vertId = 0;
					inFile.read(reinterpret_cast<char*>(&vertId), sizeof(unsigned int));

					if(needEndianSwap)
					{
						endianswap( &vertId );
					}

					indicesVec[i].push_back(vertId);
				}
			}

			primType = MESHPRIM_TYPE_TRIS;
			numVertsPrim = 3;
		}		

		return true;
	}
	else
	{
		return false;
	}
}

bool MeshFile::loadFromOBJ(const std::string& fileName, const std::string& fileNameMtlBase)
{
	// obj helper/utility vectors and structs
	typedef std::tuple<uint32_t, uint32_t, uint32_t> VertexIndexDesc;
	std::map<VertexIndexDesc, uint32_t> indexBufferDict;

	std::vector<daedalus::Vec4f> position0VecTemp;
	std::vector<daedalus::Vec4f> normals0VecTemp;
	std::vector<daedalus::Vec2f> uvcoord0VecTemp;

	std::ifstream inFile(fileName.c_str());
	if( inFile.is_open() )
	{
		std::string mtllibStr;
		std::string currMtlStr = "currMtl";

		std::string fileLineStr;
		while(getline(inFile, fileLineStr))
		{
			std::stringstream lineStream(fileLineStr);

			if(fileLineStr.size() != 0) // not a blank line
			{
				std::string objCommand;
				lineStream >> objCommand;

				if(objCommand == "v") // vertex position
				{
					Vec4f vert;

					lineStream >> vert.x;
					lineStream >> vert.y;
					lineStream >> vert.z;
					vert.w = 1.0f;

					position0VecTemp.push_back(vert);
					//usemtlVec.push_back(currMtlStr);
				}
				else if(objCommand == "vt") // vertex uvcoord
				{
					//
				}
				else if(objCommand == "vn") // vertex normal
				{
					Vec4f normal;

					lineStream >> normal.x;
					lineStream >> normal.y;
					lineStream >> normal.z;
					normal.w = 0.0f;

					normals0VecTemp.push_back(normal);
				}
				else if(objCommand == "f") // face assembly
				{
					size_t currIndex = indicesVec.size();
					indicesVec.push_back( std::vector<unsigned int>() );

					std::string faceDesc;
					while( lineStream >> faceDesc )
					{
						std::stringstream faceDescSS(faceDesc);

						std::vector<std::string> faceVec;
						std::string faceInfo;
						while(std::getline(faceDescSS, faceInfo, '/'))
						{
							faceVec.push_back(faceInfo);
						}

						int vertexId = -1;
						int texcoordId = -1;
						int normalId = -1;

						if(faceVec.size() == 1)
						{
							// f v1 v2 ... vn

							std::stringstream ssVert(faceVec[0]);
							ssVert >> vertexId;
						}
						else if(faceVec.size() == 2)
						{
							// f v1/t1 v2/t2 ... vn/tn

							std::stringstream ssVert(faceVec[0]);
							ssVert >> vertexId;

							std::stringstream ssTexCord(faceVec[1]);
							ssTexCord >> texcoordId;
						}
						else if(faceVec.size() == 3)
						{
							// f v1//n1 v2//n2 ... vn//nn
							// f v1/t1/n1 v2/t2/n2 ... vn/tn/nn

							std::stringstream ssVert(faceVec[0]);
							ssVert >> vertexId;

							if(faceVec[1] != "")
							{
								std::stringstream ssTexCord(faceVec[1]);
								ssTexCord >> texcoordId;
							}

							std::stringstream ssNorm(faceVec[2]);
							ssNorm >> normalId;
						}

						if(vertexId < 0)
						{
							// negative indices are specified relative to the current maximum vertex position

							int currVert = (int)position0Vec.size();
							vertexId = currVert + vertexId + 1;
						}

						{
							VertexIndexDesc vertex = std::make_tuple(vertexId, texcoordId, normalId);

							if (indexBufferDict.count(vertex) == 0)
							{
								indicesVec[currIndex].emplace_back(indexBufferDict.size());
								indexBufferDict[vertex] = indexBufferDict.size();

								position0Vec.emplace_back(position0VecTemp[vertexId-1]);
								normals0Vec.emplace_back(normals0VecTemp[normalId-1]);
							}
							else
							{
								indicesVec[currIndex].emplace_back(indexBufferDict.find(vertex)->second);
							}
						}
					}
				}
				else if(objCommand == "mtllib") // mtl file
				{
					lineStream >> mtllibStr;
				}
				else if(objCommand == "usemtl") // indicates that all subsequence faces should be rendered with this material, until a new material is invoked.
				{
					lineStream >> currMtlStr;
				}
			}
		}

		numVertices = (unsigned int)position0Vec.size();
		numPrimitives = (unsigned int)indicesVec.size();

		// TODO - general polygons (!!!)

		bool onlyTris = true;
		bool onlyQuads = true;
		for(size_t i = 0; i < indicesVec.size(); ++i)
		{
			if(indicesVec[i].size() != 3)
			{
				onlyTris = false;
			}

			if(indicesVec[i].size() != 4)
			{
				onlyQuads = false;
			}
		}

		if(onlyTris)
		{
			primType = MESHPRIM_TYPE_TRIS;
			numVertsPrim = 3;
		}
		else if(onlyQuads)
		{
			primType = MESHPRIM_TYPE_QUADS;
			numVertsPrim = 4;
		}
		else
		{
			triangulate();
		}
		
		if(mtllibStr != "")
		{
			std::string fileNameMtlFull;
			std::string fileNameMtlFull0 = "assets/" + mtllibStr;
			std::string fileNameMtlFull1 = "assets/" + fileName.substr(0, fileName.length()-3) + "mtl";

			if( findFullPath(fileNameMtlFull0) )
			{
				fileNameMtlFull = fileNameMtlFull0;
			}
			else if( findFullPath(fileNameMtlFull1) )
			{
				fileNameMtlFull = fileNameMtlFull1;
			}

			if(fileNameMtlFull == "")
			{
				std::cout << "[ERR] MESHMTLFILE OPEN - " << mtllibStr << std::endl;
				return 1;
			}

			std::ifstream inFileMtl(fileNameMtlFull.c_str());
			if( inFile.is_open() )
			{
				std::string currMtlStr = "currMtl";
				MaterialDescObj matDesc;
				matDesc.name = currMtlStr;

				std::string fileLineStr;
				while(getline(inFileMtl, fileLineStr))
				{
					std::stringstream lineStream(fileLineStr);

					if(fileLineStr.size() != 0) // not a blank line
					{
						std::string objCommand;
						lineStream >> objCommand;

						if(objCommand == "newmtl") // material definition
						{
							materialVec.push_back(matDesc);

							lineStream >> currMtlStr;

							matDesc.name = currMtlStr;
						}
						else if(objCommand == "Kd") // diffuse term
						{
							Vec4f kd;

							lineStream >> kd.x;
							lineStream >> kd.y;
							lineStream >> kd.z;
							kd.w = 1.0f;

							matDesc.kd = kd;
						}
					}
				}
				materialVec.push_back(matDesc);
			}

			for(size_t i = 0; i < position0Vec.size(); ++i)
			{
				std::string mtlName = usemtlVec[i];
				bool hasFound = false;

				for(unsigned int m = 0; m < position0Vec.size(); ++m)
				{
					if( materialVec[m].name == mtlName )
					{
						colors0Vec.push_back( materialVec[m].kd );
						hasFound = true;
						break;
					}
				}

				if(!hasFound)
				{
					colors0Vec.push_back( Vec4f(1.0f, 1.0f, 1.0f, 1.0f) );
				}
			}

			materialVec.clear();
		}

		return true;
	}
	else
	{
		return false;
	}
}

void MeshFile::createFSQPos()
{
	std::stringstream ssName;
	ssName << "fsq_pos";

	coreName = ssName.str();
	extName = "gen";

	primType = MESHPRIM_TYPE_TRIS;
	numVertsPrim = 3;

	numVertices = 6;
	numPrimitives = 2;

	position0Vec.push_back(Vec4f(-1.0f, -1.0f, -1.0f, 1.0f));
	position0Vec.push_back(Vec4f( 1.0f, -1.0f, -1.0f, 1.0f));
	position0Vec.push_back(Vec4f( 1.0f,  1.0f, -1.0f, 1.0f));
	position0Vec.push_back(Vec4f( 1.0f,  1.0f, -1.0f, 1.0f));
	position0Vec.push_back(Vec4f(-1.0f,  1.0f, -1.0f, 1.0f));
	position0Vec.push_back(Vec4f(-1.0f, -1.0f, -1.0f, 1.0f));

	fillAttributeVec();
	printInfo();
}

void MeshFile::createFSQPosTex()
{
	std::stringstream ssName;
	ssName << "fsq_postex";

	coreName = ssName.str();
	extName = "gen";

	primType = MESHPRIM_TYPE_TRIS;
	numVertsPrim = 3;

	numVertices = 6;
	numPrimitives = 2;

	position0Vec.push_back(Vec4f(-1.0f, -1.0f, -1.0f, 1.0f));
	position0Vec.push_back(Vec4f( 1.0f, -1.0f, -1.0f, 1.0f));
	position0Vec.push_back(Vec4f( 1.0f,  1.0f, -1.0f, 1.0f));
	
	position0Vec.push_back(Vec4f( 1.0f,  1.0f, -1.0f, 1.0f));
	position0Vec.push_back(Vec4f(-1.0f,  1.0f, -1.0f, 1.0f));
	position0Vec.push_back(Vec4f(-1.0f, -1.0f, -1.0f, 1.0f));

	uvcoord0Vec.push_back(Vec2f(0.0f,1.0f));
	uvcoord0Vec.push_back(Vec2f(1.0f,1.0f));
	uvcoord0Vec.push_back(Vec2f(1.0f,0.0f));

	uvcoord0Vec.push_back(Vec2f(1.0f,0.0f));
	uvcoord0Vec.push_back(Vec2f(0.0f,0.0f));
	uvcoord0Vec.push_back(Vec2f(0.0f,1.0f));

	fillAttributeVec();
	printInfo();
}

void MeshFile::createWorldAxis(float scale)
{
	std::stringstream ssName;
	ssName << "worldaxis";

	coreName = ssName.str();
	extName = "gen";

	primType = MESHPRIM_TYPE_LINES;
	numVertsPrim = 2;

	numVertices = 6;
	numPrimitives = 3;

	position0Vec.push_back( Vec4f(0.0f,0.0f,0.0f,1.0f) );
	position0Vec.push_back( Vec4f(scale,0.0f,0.0f,1.0f) );

	position0Vec.push_back( Vec4f(0.0f,0.0f,0.0f,1.0f) );
	position0Vec.push_back( Vec4f(0.0f,scale,0.0f,1.0f) );

	position0Vec.push_back( Vec4f(0.0f,0.0f,0.0f,1.0f) );
	position0Vec.push_back( Vec4f(0.0f,0.0f,scale,1.0f) );

	colors0Vec.push_back( Vec4f(1.0f,0.0f,0.0f,1.0f) );
	colors0Vec.push_back( Vec4f(1.0f,0.0f,0.0f,1.0f) );
	colors0Vec.push_back( Vec4f(0.0f,1.0f,0.0f,1.0f) );
	colors0Vec.push_back( Vec4f(0.0f,1.0f,0.0f,1.0f) );
	colors0Vec.push_back( Vec4f(0.0f,0.0f,1.0f,1.0f) );
	colors0Vec.push_back( Vec4f(0.0f,0.0f,1.0f,1.0f) );

	fillAttributeVec();
	printInfo();
}

void MeshFile::createGridTris(float w, float l, unsigned int wSegs, unsigned int lSegs)
{
	std::stringstream ssName;
	ssName << "gridTris_" << wSegs << "x" << lSegs;

	coreName = ssName.str();
	extName = "gen";

	primType = MESHPRIM_TYPE_TRIS;
	numVertsPrim = 3;

	numVertices = (wSegs + 1) * (lSegs + 1);
	numPrimitives = wSegs * lSegs * 2;

	float dw = w / wSegs;
	float dl = l / lSegs;

	for(unsigned int j = 0; j <= lSegs; ++j)
	{
		for(unsigned int i = 0; i <= wSegs; ++i)
		{
			position0Vec.push_back(Vec4f(-0.5f*w+i*dw, 0.0f, 0.5f*l-j*dl, 1.0f));
			normals0Vec.push_back(Vec4f(0.0f,1.0f,0.0f,0.0f));
			uvcoord0Vec.push_back(Vec2f((float)i/wSegs, (float)j/lSegs));
		}
	}

	for(unsigned int j = 0; j < lSegs; ++j)
	{
		for(unsigned int i = 0; i < wSegs; ++i)
		{
			unsigned int lowerLeft = j * (wSegs + 1) + i;
			unsigned int lowerRight = j * (wSegs + 1) + i + 1;
			unsigned int topRight = (j + 1) * (wSegs + 1) + i + 1;
			unsigned int topLeft = (j + 1) * (wSegs + 1) + i;
			
			std::vector< unsigned int > tri0;
			std::vector< unsigned int > tri1;

			tri0.push_back(lowerLeft);
			tri0.push_back(lowerRight);
			tri0.push_back(topRight);

			tri1.push_back(lowerLeft);
			tri1.push_back(topRight);
			tri1.push_back(topLeft);

			indicesVec.push_back(tri0);
			indicesVec.push_back(tri1);
		}
	}

	fillAttributeVec();
	printInfo();
}

void MeshFile::createGridWire(float w, float l, unsigned int wSegs, unsigned int lSegs)
{
	std::stringstream ssName;
	ssName << "gridWire_" << wSegs << "x" << lSegs;

	coreName = ssName.str();
	extName = "gen";

	primType = MESHPRIM_TYPE_LINES;
	numVertsPrim = 2;

	numVertices = 2*(wSegs+1) + 2*(lSegs+1);
	numPrimitives = (wSegs+1) + (lSegs+1);

	Vec4f iStep = Vec4f(w/wSegs, 0.0f, 0.0f, 0.0f);
	Vec4f jStep = Vec4f(0.0f, 0.0f, l/lSegs, 0.0f);

	for(uint i = 0; i <= wSegs; ++i)
	{
		Vec4f curr0 = Vec4f(-w * 0.5f, 0.0f, -l * 0.5f, 1.0f) + (float)i*iStep;
		position0Vec.push_back( curr0 );

		Vec4f curr1 = Vec4f(-w * 0.5f, 0.0f,  l * 0.5f, 1.0f) + (float)i*iStep;
		position0Vec.push_back( curr1 );
	}

	for(uint j = 0; j <= lSegs; ++j)
	{
		Vec4f curr0 = Vec4f(-w * 0.5f, 0.0f, -l * 0.5f, 1.0f) + (float)j*jStep;
		position0Vec.push_back( curr0 );

		Vec4f curr1 = Vec4f( w * 0.5f, 0.0f, -l * 0.5f, 1.0f) + (float)j*jStep;
		position0Vec.push_back( curr1 );
	}

	fillAttributeVec();
	printInfo();
}

void MeshFile::createBoxTris(float w, float h, float l)
{
	std::stringstream ssName;
	ssName << "boxTris_" << 1 << "x" << 1 << "x" << 1;

	coreName = ssName.str();
	extName = "gen";

	primType = MESHPRIM_TYPE_TRIS;
	numVertsPrim = 3;

	numVertices = 6*4;
	numPrimitives = 6*2;

	w *= 0.5f;
	h *= 0.5f;
	l *= 0.5f;

	// bottom face
	position0Vec.push_back( Vec4f( w, -h,  l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f, -1.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(0.0f, 0.0f) );
	position0Vec.push_back( Vec4f(-w, -h,  l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f, -1.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(1.0f, 0.0f) );
	position0Vec.push_back( Vec4f(-w, -h, -l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f, -1.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(1.0f, 1.0f) );
	position0Vec.push_back( Vec4f( w, -h, -l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f, -1.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(0.0f, 1.0f) );

	// top face
	position0Vec.push_back( Vec4f(-w,  h,  l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f,  1.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(0.0f, 0.0f) );
	position0Vec.push_back( Vec4f( w,  h,  l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f,  1.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(1.0f, 0.0f) );
	position0Vec.push_back( Vec4f( w,  h, -l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f,  1.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(1.0f, 1.0f) );
	position0Vec.push_back( Vec4f(-w,  h, -l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f,  1.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(0.0f, 1.0f) );

	// left face
	position0Vec.push_back( Vec4f(-w, -h, -l, 1.0f) ); normals0Vec.push_back( Vec4f(-1.0f,  0.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(0.0f, 0.0f) );
	position0Vec.push_back( Vec4f(-w, -h,  l, 1.0f) ); normals0Vec.push_back( Vec4f(-1.0f,  0.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(1.0f, 0.0f) );
	position0Vec.push_back( Vec4f(-w,  h,  l, 1.0f) ); normals0Vec.push_back( Vec4f(-1.0f,  0.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(1.0f, 1.0f) );
	position0Vec.push_back( Vec4f(-w,  h, -l, 1.0f) ); normals0Vec.push_back( Vec4f(-1.0f,  0.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(0.0f, 1.0f) );

	// right face
	position0Vec.push_back( Vec4f( w, -h,  l, 1.0f) ); normals0Vec.push_back( Vec4f( 1.0f,  0.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(0.0f, 0.0f) );
	position0Vec.push_back( Vec4f( w, -h, -l, 1.0f) ); normals0Vec.push_back( Vec4f( 1.0f,  0.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(1.0f, 0.0f) );
	position0Vec.push_back( Vec4f( w,  h, -l, 1.0f) ); normals0Vec.push_back( Vec4f( 1.0f,  0.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(1.0f, 1.0f) );
	position0Vec.push_back( Vec4f( w,  h,  l, 1.0f) ); normals0Vec.push_back( Vec4f( 1.0f,  0.0f,  0.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(0.0f, 1.0f) );

	// front face
	position0Vec.push_back( Vec4f(-w, -h,  l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f,  0.0f,  1.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(0.0f, 0.0f) );
	position0Vec.push_back( Vec4f( w, -h,  l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f,  0.0f,  1.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(1.0f, 0.0f) );
	position0Vec.push_back( Vec4f( w,  h,  l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f,  0.0f,  1.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(1.0f, 1.0f) );
	position0Vec.push_back( Vec4f(-w,  h,  l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f,  0.0f,  1.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(0.0f, 1.0f) );

	// back face
	position0Vec.push_back( Vec4f( w, -h, -l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f,  0.0f, -1.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(0.0f, 0.0f) );
	position0Vec.push_back( Vec4f(-w, -h, -l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f,  0.0f, -1.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(1.0f, 0.0f) );
	position0Vec.push_back( Vec4f(-w,  h, -l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f,  0.0f, -1.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(1.0f, 1.0f) );
	position0Vec.push_back( Vec4f( w,  h, -l, 1.0f) ); normals0Vec.push_back( Vec4f( 0.0f,  0.0f, -1.0f, 0.0f) ); uvcoord0Vec.push_back( Vec2f(0.0f, 1.0f) );

	std::vector< unsigned int > tri00;
	std::vector< unsigned int > tri01;
	tri00.push_back( 0); tri00.push_back( 1); tri00.push_back( 2);
	tri01.push_back( 0); tri01.push_back( 2); tri01.push_back( 3);

	std::vector< unsigned int > tri02;
	std::vector< unsigned int > tri03;
	tri02.push_back( 4); tri02.push_back( 5); tri02.push_back( 6);
	tri03.push_back( 4); tri03.push_back( 6); tri03.push_back( 7);

	std::vector< unsigned int > tri04;
	std::vector< unsigned int > tri05;
	tri04.push_back( 8); tri04.push_back( 9); tri04.push_back(10);
	tri05.push_back( 8); tri05.push_back(10); tri05.push_back(11);

	std::vector< unsigned int > tri06;
	std::vector< unsigned int > tri07;
	tri06.push_back(12); tri06.push_back(13); tri06.push_back(14);
	tri07.push_back(12); tri07.push_back(14); tri07.push_back(15);

	std::vector< unsigned int > tri08;
	std::vector< unsigned int > tri09;
	tri08.push_back(16); tri08.push_back(17); tri08.push_back(18);
	tri09.push_back(16); tri09.push_back(18); tri09.push_back(19);

	std::vector< unsigned int > tri10;
	std::vector< unsigned int > tri11;
	tri10.push_back(20); tri10.push_back(21); tri10.push_back(22);
	tri11.push_back(20); tri11.push_back(22); tri11.push_back(23);

	indicesVec.push_back(tri00);
	indicesVec.push_back(tri01);
	indicesVec.push_back(tri02);
	indicesVec.push_back(tri03);
	indicesVec.push_back(tri04);
	indicesVec.push_back(tri05);
	indicesVec.push_back(tri06);
	indicesVec.push_back(tri07);
	indicesVec.push_back(tri08);
	indicesVec.push_back(tri09);
	indicesVec.push_back(tri10);
	indicesVec.push_back(tri11);
	
	fillAttributeVec();
	printInfo();
}

void MeshFile::createBoxWire(float w, float h, float l)
{
	std::stringstream ssName;
	ssName << "boxWire_" << 1 << "x" << 1 << "x" << 1;

	coreName = ssName.str();
	extName = "gen";

	primType = MESHPRIM_TYPE_LINES;
	numVertsPrim = 2;

	numVertices = 8;
	numPrimitives = 12;

	w *= 0.5f;
	h *= 0.5f;
	l *= 0.5f;

	// bottom face
	position0Vec.push_back( Vec4f( w, -h,  l, 1.0f) );
	position0Vec.push_back( Vec4f(-w, -h,  l, 1.0f) );
	position0Vec.push_back( Vec4f(-w, -h, -l, 1.0f) );
	position0Vec.push_back( Vec4f( w, -h, -l, 1.0f) );

	// top face
	position0Vec.push_back( Vec4f( w,  h,  l, 1.0f) );
	position0Vec.push_back( Vec4f(-w,  h,  l, 1.0f) );
	position0Vec.push_back( Vec4f(-w,  h, -l, 1.0f) );
	position0Vec.push_back( Vec4f( w,  h, -l, 1.0f) );

	std::vector< unsigned int > line00;
	std::vector< unsigned int > line01;
	std::vector< unsigned int > line02;
	std::vector< unsigned int > line03;
	line00.push_back(0); line00.push_back(1);
	line01.push_back(1); line01.push_back(2);
	line02.push_back(2); line02.push_back(3);
	line03.push_back(3); line03.push_back(0);

	std::vector< unsigned int > line04;
	std::vector< unsigned int > line05;
	std::vector< unsigned int > line06;
	std::vector< unsigned int > line07;
	line04.push_back(4); line04.push_back(5);
	line05.push_back(5); line05.push_back(6);
	line06.push_back(6); line06.push_back(7);
	line07.push_back(7); line07.push_back(4);

	std::vector< unsigned int > line08;
	std::vector< unsigned int > line09;
	std::vector< unsigned int > line10;
	std::vector< unsigned int > line11;
	line08.push_back(0); line08.push_back(0+4);
	line09.push_back(1); line09.push_back(1+4);
	line10.push_back(2); line10.push_back(2+4);
	line11.push_back(3); line11.push_back(3+4);

	indicesVec.push_back(line00);
	indicesVec.push_back(line01);
	indicesVec.push_back(line02);
	indicesVec.push_back(line03);
	indicesVec.push_back(line04);
	indicesVec.push_back(line05);
	indicesVec.push_back(line06);
	indicesVec.push_back(line07);
	indicesVec.push_back(line08);
	indicesVec.push_back(line09);
	indicesVec.push_back(line10);
	indicesVec.push_back(line11);
	
	fillAttributeVec();
	printInfo();
}

void MeshFile::createBoxWire(float minX, float minY, float minZ, float maxX, float maxY, float maxZ)
{
	std::stringstream ssName;
	ssName << "boxWire_" << 1 << "x" << 1 << "x" << 1;

	coreName = ssName.str();
	extName = "gen";

	primType = MESHPRIM_TYPE_LINES;
	numVertsPrim = 2;

	numVertices = 8;
	numPrimitives = 12;

	// bottom face
	position0Vec.push_back(Vec4f(maxX, minY, maxZ, 1.0f));
	position0Vec.push_back(Vec4f(minX, minY, maxZ, 1.0f));
	position0Vec.push_back(Vec4f(minX, minY, minZ, 1.0f));
	position0Vec.push_back(Vec4f(maxX, minY, minZ, 1.0f));

	// top face
	position0Vec.push_back(Vec4f(maxX, maxY, maxZ, 1.0f));
	position0Vec.push_back(Vec4f(minX, maxY, maxZ, 1.0f));
	position0Vec.push_back(Vec4f(minX, maxY, minZ, 1.0f));
	position0Vec.push_back(Vec4f(maxX, maxY, minZ, 1.0f));

	std::vector< unsigned int > line00;
	std::vector< unsigned int > line01;
	std::vector< unsigned int > line02;
	std::vector< unsigned int > line03;
	line00.push_back(0); line00.push_back(1);
	line01.push_back(1); line01.push_back(2);
	line02.push_back(2); line02.push_back(3);
	line03.push_back(3); line03.push_back(0);

	std::vector< unsigned int > line04;
	std::vector< unsigned int > line05;
	std::vector< unsigned int > line06;
	std::vector< unsigned int > line07;
	line04.push_back(4); line04.push_back(5);
	line05.push_back(5); line05.push_back(6);
	line06.push_back(6); line06.push_back(7);
	line07.push_back(7); line07.push_back(4);

	std::vector< unsigned int > line08;
	std::vector< unsigned int > line09;
	std::vector< unsigned int > line10;
	std::vector< unsigned int > line11;
	line08.push_back(0); line08.push_back(0 + 4);
	line09.push_back(1); line09.push_back(1 + 4);
	line10.push_back(2); line10.push_back(2 + 4);
	line11.push_back(3); line11.push_back(3 + 4);

	indicesVec.push_back(line00);
	indicesVec.push_back(line01);
	indicesVec.push_back(line02);
	indicesVec.push_back(line03);
	indicesVec.push_back(line04);
	indicesVec.push_back(line05);
	indicesVec.push_back(line06);
	indicesVec.push_back(line07);
	indicesVec.push_back(line08);
	indicesVec.push_back(line09);
	indicesVec.push_back(line10);
	indicesVec.push_back(line11);

	fillAttributeVec();
	printInfo();
}

void MeshFile::createSphereTris(float rad, unsigned int hSegs, unsigned int vSegs)
{
	std::stringstream ssName;
	ssName << "sphereTris_" << hSegs << "x" << vSegs;

	coreName = ssName.str();
	extName = "gen";

	primType = MESHPRIM_TYPE_TRIS;
	numVertsPrim = 3;

	numVertices = (hSegs + 1) * (vSegs + 1);
	numPrimitives = hSegs * vSegs * 2;

	float dphi = (float)two_pi / (float)hSegs;
	float dtheta = (float)pi / (float)vSegs;

	for(unsigned int v = 0; v <= vSegs; ++v)
	{
		float theta = v * dtheta;

		for(unsigned int h = 0; h <= hSegs; ++h)
		{
			float phi = h * dphi;

			float x = sin(theta) * cos(phi);
			float y = cos(theta);
			float z = sin(theta) * sin(phi);

			position0Vec.push_back( Vec4f(rad*x,rad*y,rad*z,1.0f) );
			
			normals0Vec.push_back( Vec4f(x,y,z,0.0f) );

			uvcoord0Vec.push_back( Vec2f(1.0f-(float)h/hSegs, (float)v/vSegs) );
		}
	}

	for(unsigned int v = 0; v < vSegs; v++)
	{
		for(unsigned int h = 0; h < hSegs; h++)
		{
			unsigned int topRight = v * (hSegs + 1) + h;
			unsigned int topLeft = v * (hSegs + 1) + h + 1;
			unsigned int lowerRight = (v + 1) * (hSegs + 1) + h;
			unsigned int lowerLeft = (v + 1) * (hSegs + 1) + h + 1;

			std::vector< unsigned int > tri0;
			std::vector< unsigned int > tri1;

			tri0.push_back(lowerLeft);
			tri0.push_back(lowerRight);
			tri0.push_back(topRight);

			tri1.push_back(lowerLeft);
			tri1.push_back(topRight);
			tri1.push_back(topLeft);

			indicesVec.push_back(tri0);
			indicesVec.push_back(tri1);
		}
	}

	fillAttributeVec();
	printInfo();
}

void MeshFile::createSphereWire(float rad, unsigned int segs)
{
	std::stringstream ssName;
	ssName << "sphereWire_" << segs;

	coreName = ssName.str();
	extName = "gen";

	primType = MESHPRIM_TYPE_LINES;
	numVertsPrim = 2;

	numVertices = 3*(segs*2)+3*2;
	numPrimitives = 3*segs+3;

	float dalpha = (float)two_pi / (float)segs;

	// circle in YZ plane
	for(unsigned int i = 0; i < segs; ++i)
	{
		float alpha0 = i * dalpha;
		float alpha1 = (i+1) * dalpha;

		float x0 = 0.0f;
		float y0 = sin(alpha0);
		float z0 = cos(alpha0);

		float x1 = 0.0f;
		float y1 = sin(alpha1);
		float z1 = cos(alpha1);

		position0Vec.push_back(Vec4f(rad * x0, rad * y0, rad * z0, 1.0f));
		position0Vec.push_back(Vec4f(rad * x1, rad * y1, rad * z1, 1.0f));
	}

	// circle in XZ plane
	for(unsigned int i = 0; i < segs; ++i)
	{
		float alpha0 = i * dalpha;
		float alpha1 = (i+1) * dalpha;

		float x0 = cos(alpha0);
		float y0 = 0.0f;
		float z0 = sin(alpha0);

		float x1 = cos(alpha1);
		float y1 = 0.0f;
		float z1 = sin(alpha1);

		position0Vec.push_back(Vec4f(rad * x0, rad * y0, rad * z0, 1.0f));
		position0Vec.push_back(Vec4f(rad * x1, rad * y1, rad * z1, 1.0f));
	}

	// circle in XY plane
	for(unsigned int i = 0; i < segs; ++i)
	{
		float alpha0 = i * dalpha;
		float alpha1 = (i+1) * dalpha;

		float x0 = cos(alpha0);
		float y0 = sin(alpha0);
		float z0 = 0.0f;

		float x1 = cos(alpha1);
		float y1 = sin(alpha1);
		float z1 = 0.0f;

		position0Vec.push_back(Vec4f(rad * x0, rad * y0, rad * z0, 1.0f));
		position0Vec.push_back(Vec4f(rad * x1, rad * y1, rad * z1, 1.0f));
	}

	// small cross at sphere origin
	float crossHalf = rad*0.1f;

	position0Vec.push_back(Vec4f(-crossHalf, 0.0f, 0.0f, 1.0f));
	position0Vec.push_back(Vec4f( crossHalf, 0.0f, 0.0f, 1.0f));

	position0Vec.push_back(Vec4f(0.0f, -crossHalf, 0.0f, 1.0f));
	position0Vec.push_back(Vec4f(0.0f,  crossHalf, 0.0f, 1.0f));

	position0Vec.push_back(Vec4f(0.0f, 0.0f, -crossHalf, 1.0f));
	position0Vec.push_back(Vec4f(0.0f, 0.0f,  crossHalf, 1.0f));

	fillAttributeVec();
	printInfo();
}

void MeshFile::createConeWire(float height, float angle, unsigned int segs)
{
	std::stringstream ssName;
	ssName << "coneWire_" << segs;

	coreName = ssName.str();
	extName = "gen";

	primType = MESHPRIM_TYPE_LINES;
	numVertsPrim = 2;

	numVertices = segs*2 + 4*2;
	numPrimitives = segs + 4;

	float dalpha = (float)two_pi / (float)segs;
	float radius = height * tan(deg2rad(angle));

	// circle in XZ plane shifted with minus height
	for(unsigned int i = 0; i < segs; i++)
	{
		float alpha0 = i * dalpha;
		float alpha1 = (i+1) * dalpha;

		float x0 = cos(alpha0);
		float y0 = -height;
		float z0 = sin(alpha0);

		float x1 = cos(alpha1);
		float y1 = -height;
		float z1 = sin(alpha1);

		position0Vec.push_back(Vec4f(radius * x0, y0, radius * z0, 1.0f));
		position0Vec.push_back(Vec4f(radius * x1, y1, radius * z1, 1.0f));
	}

	// 4 sticks
	float dphi = (float)two_pi / 4.0f;

	position0Vec.push_back(Vec4f(0.0f, 0.0f, 0.0f, 1.0f));
	position0Vec.push_back(Vec4f(radius * cos(dphi*0), -height, radius * sin(dphi*0), 1.0f));

	position0Vec.push_back(Vec4f(0.0f, 0.0f, 0.0f, 1.0f));
	position0Vec.push_back(Vec4f(radius * cos(dphi*1), -height, radius * sin(dphi*1), 1.0f));

	position0Vec.push_back(Vec4f(0.0f, 0.0f, 0.0f, 1.0f));
	position0Vec.push_back(Vec4f(radius * cos(dphi*2), -height,  radius * sin(dphi*2), 1.0f));

	position0Vec.push_back(Vec4f(0.0f, 0.0f, 0.0f, 1.0f));
	position0Vec.push_back(Vec4f(radius * cos(dphi*3), -height,  radius * sin(dphi*3), 1.0f));

	fillAttributeVec();
	printInfo();
}

void MeshFile::fillAttributeVec()
{
	hasAttribute[0] = position0Vec.size() != 0;
	hasAttribute[1] = position1Vec.size() != 0;
	hasAttribute[2] = normals0Vec.size() != 0;
	hasAttribute[3] = normals1Vec.size() != 0;
	hasAttribute[4] = colors0Vec.size() != 0;
	hasAttribute[5] = colors1Vec.size() != 0;
	hasAttribute[6] = uvcoord0Vec.size() != 0;
	hasAttribute[7] = uvcoord1Vec.size() != 0;
}

AABB4f MeshFile::getAABB() const
{
	AABB4f meshAABB(False);

	for(unsigned int i = 0; i < position0Vec.size(); ++i)
	{
		meshAABB += position0Vec[i];
	}

	return meshAABB;
}

void MeshFile::generateNormals()
{
	normals0Vec.clear();
	normals0Vec.resize( position0Vec.size(), Vec4f(zero) );

	hasAttribute[2] = true;

	if(primType == MESHPRIM_TYPE_TRIS)
	{
		for(unsigned int i = 0; i < indicesVec.size(); ++i)
		{
			Vec4f v0 = position0Vec[ indicesVec[i][0] ];
			Vec4f v1 = position0Vec[ indicesVec[i][1] ];
			Vec4f v2 = position0Vec[ indicesVec[i][2] ];

			Vec4f plane = createPlane<float>(v0,v1,v2);
			plane.w = 0.0f;

			normals0Vec[ indicesVec[i][0] ] += plane;
			normals0Vec[ indicesVec[i][1] ] += plane;
			normals0Vec[ indicesVec[i][2] ] += plane;
		}
	}
	else if(primType == MESHPRIM_TYPE_QUADS)
	{
		for(unsigned int i = 0; i < indicesVec.size(); ++i)
		{
			Vec4f v0 = position0Vec[ indicesVec[i][0] ];
			Vec4f v1 = position0Vec[ indicesVec[i][1] ];
			Vec4f v2 = position0Vec[ indicesVec[i][2] ];

			Vec4f plane = createPlane<float>(v0,v1,v2);
			plane.w = 0.0f;

			normals0Vec[ indicesVec[i][0] ] += plane;
			normals0Vec[ indicesVec[i][1] ] += plane;
			normals0Vec[ indicesVec[i][2] ] += plane;
			normals0Vec[ indicesVec[i][3] ] += plane;
		}
	}

	for(unsigned int i = 0; i < normals0Vec.size(); ++i)
	{
		normals0Vec[i] = normalize( normals0Vec[i] );
	}
}

void MeshFile::copyNormalsToColors()
{
	colors0Vec.clear();
	colors0Vec.resize( normals0Vec.size(), Vec4f(zero) );
	hasAttribute[4] = true;

	for(unsigned int i = 0; i < normals0Vec.size(); ++i)
	{
		colors0Vec[i] = abs( normals0Vec[i] );
		colors0Vec[i].w = 1.0f;
	}

	normals0Vec.clear();
	hasAttribute[2] = false;
}

void MeshFile::setPositionVec(const Vec4f* x, unsigned int numElements)
{
	position0Vec.clear();
	position0Vec.resize( numElements );

	for(size_t i = 0; i < numElements; ++i)
	{
		position0Vec[i] = x[i];
	}

	// TODO!!!
	primType = MESHPRIM_TYPE_POINTS;
	numVertices = numElements;
	numPrimitives = numElements;

	hasAttribute[0] = true;
}

void MeshFile::setNormalVec(const Vec4f* x, unsigned int numElements)
{
	normals0Vec.clear();
	normals0Vec.resize( numElements );

	for(size_t i = 0; i < numElements; ++i)
	{
		normals0Vec[i] = x[i];
	}

	hasAttribute[2] = true;
}

void MeshFile::setColorVec(const Vec4f* x, unsigned int numElements)
{
	colors0Vec.clear();
	colors0Vec.resize( numElements );

	for(size_t i = 0; i < numElements; ++i)
	{
		colors0Vec[i] = x[i];
	}

	hasAttribute[4] = true;
}

void MeshFile::setColorVec(float r, float g, float b)
{
	colors0Vec.clear();
	colors0Vec.resize( position0Vec.size(), Vec4f(r,g,b,1.0f) );

	hasAttribute[4] = true;
}

void MeshFile::add(const MeshFile& meshdata, const Mat44f& mat)
{
	// only merge same type - quads qith quads, tris with tris
	// FIX triangulate()
	if (primType == MESHPRIM_TYPE_NOTSET)
	{
		primType = meshdata.primType;
		numVertsPrim = meshdata.numVertsPrim;
	}

	std::vector< Vec4f > position0VecToAdd = meshdata.position0Vec;
	for(unsigned int i = 0; i < position0VecToAdd.size(); ++i)
	{
		position0VecToAdd[i] = position0VecToAdd[i] * mat;
	}
	position0Vec.insert( position0Vec.end(), position0VecToAdd.begin(), position0VecToAdd.end() );

	std::vector< Vec4f > normals0VecToAdd = meshdata.normals0Vec;
	for(unsigned int i = 0; i < normals0VecToAdd.size(); ++i)
	{
		normals0VecToAdd[i] = normals0VecToAdd[i] * mat;
	}
	normals0Vec.insert( normals0Vec.end(), normals0VecToAdd.begin(), normals0VecToAdd.end() );

	colors0Vec.insert( colors0Vec.end(), meshdata.colors0Vec.begin(), meshdata.colors0Vec.end() );
	uvcoord0Vec.insert( uvcoord0Vec.end(), meshdata.uvcoord0Vec.begin(), meshdata.uvcoord0Vec.end() );

	std::vector< unsigned int > indicesVecFlatToAdd = meshdata.indicesVecFlat;
	for(unsigned int i = 0; i < indicesVecFlatToAdd.size(); ++i)
	{
		indicesVecFlatToAdd[i] = indicesVecFlatToAdd[i] + (unsigned int)numVertices;
	}
	indicesVecFlat.insert( indicesVecFlat.end(), indicesVecFlatToAdd.begin(), indicesVecFlatToAdd.end() );

	numVertices = (unsigned int)position0Vec.size();
	numPrimitives = (unsigned int)indicesVecFlat.size() / numVertsPrim;

	fillAttributeVec();
}

void MeshFile::addParallelogram(const Vec4f& anchor, const Vec4f& offset1, const Vec4f& offset2, const Vec4f& c)
{
	coreName = "cornellQuads";
	extName = "gen";

	primType = MESHPRIM_TYPE_QUADS;
	numVertsPrim = 4;

	numVertices += 4;
	numPrimitives += 1;

	position0Vec.push_back( anchor );
	position0Vec.push_back( anchor+offset1 );
	position0Vec.push_back( anchor+offset1+offset2 );
	position0Vec.push_back( anchor+offset2 );
	
	//Vec4f n = normalize( cross( offset1, offset2 ) );
	//normals0Vec.push_back( n );
	//normals0Vec.push_back( n );
	//normals0Vec.push_back( n );
	//normals0Vec.push_back( n );

	colors0Vec.push_back( c );
	colors0Vec.push_back( c );
	colors0Vec.push_back( c );
	colors0Vec.push_back( c );

	std::vector< unsigned int > face;
	face.push_back( 4*(unsigned int)indicesVec.size()+0 );
	face.push_back( 4*(unsigned int)indicesVec.size()+1 );
	face.push_back( 4*(unsigned int)indicesVec.size()+2 );
	face.push_back( 4*(unsigned int)indicesVec.size()+3 );

	indicesVec.push_back( face );
	
	hasAttribute[0] = position0Vec.size() != 0;
	hasAttribute[1] = position1Vec.size() != 0;
	hasAttribute[2] = normals0Vec.size() != 0;
	hasAttribute[3] = normals1Vec.size() != 0;
	hasAttribute[4] = colors0Vec.size() != 0;
	hasAttribute[5] = colors1Vec.size() != 0;
	hasAttribute[6] = uvcoord0Vec.size() != 0;
	hasAttribute[7] = uvcoord1Vec.size() != 0;
}

void MeshFile::triangulate()
{
	if (indicesVec.size())
	{
		std::vector< std::vector<unsigned int> > indicesVecNew;

		for (size_t i = 0; i < indicesVec.size(); ++i)
		{
			if (indicesVec[i].size() == 3)
			{
				indicesVecNew.push_back(indicesVec[i]);
			}
			else if (indicesVec[i].size() == 4)
			{
				std::vector< unsigned int > tri0;
				std::vector< unsigned int > tri1;

				tri0.push_back(indicesVec[i][0]);
				tri0.push_back(indicesVec[i][1]);
				tri0.push_back(indicesVec[i][2]);

				tri1.push_back(indicesVec[i][0]);
				tri1.push_back(indicesVec[i][2]);
				tri1.push_back(indicesVec[i][3]);

				indicesVecNew.push_back(tri0);
				indicesVecNew.push_back(tri1);
			}
		}

		indicesVec = indicesVecNew;
		numPrimitives = (unsigned int)indicesVec.size();

		primType = MESHPRIM_TYPE_TRIS;
		numVertsPrim = 3;
	}

	if (indicesVecFlat.size() && primType == MESHPRIM_TYPE_QUADS)
	{
		std::vector< unsigned int > indicesVecNew;

		for (size_t i = 0; i < indicesVecFlat.size() / 4; ++i)
		{
			indicesVecNew.push_back(indicesVecFlat[4 * i] + 0);
			indicesVecNew.push_back(indicesVecFlat[4 * i] + 1);
			indicesVecNew.push_back(indicesVecFlat[4 * i] + 2);

			indicesVecNew.push_back(indicesVecFlat[4 * i] + 0);
			indicesVecNew.push_back(indicesVecFlat[4 * i] + 2);
			indicesVecNew.push_back(indicesVecFlat[4 * i] + 3);
		}

		indicesVecFlat = indicesVecNew;
		numPrimitives = (unsigned int)indicesVecNew.size() / 3;

		primType = MESHPRIM_TYPE_TRIS;
		numVertsPrim = 3;
	}
}

void MeshFile::tesselate()
{
	std::vector<Vec4f> verticesTessVec;
	std::vector<Vec4f> colorsTessVec;
	std::vector< std::vector<unsigned int> > indicesTessVec;

	if(primType == MESHPRIM_TYPE_TRIS)
	{
		for(unsigned int i = 0; i < indicesVec.size(); ++i)
		{
			Vec4f v0 = position0Vec[ indicesVec[i][0] ];
			Vec4f v1 = position0Vec[ indicesVec[i][1] ];
			Vec4f v2 = position0Vec[ indicesVec[i][2] ];

			Vec4f v01 = 0.5f * (v0+v1);
			Vec4f v12 = 0.5f * (v1+v2);
			Vec4f v20 = 0.5f * (v2+v0);

			Vec4f c0 = colors0Vec[ indicesVec[i][0] ];
			Vec4f c1 = colors0Vec[ indicesVec[i][1] ];
			Vec4f c2 = colors0Vec[ indicesVec[i][2] ];

			Vec4f c01 = 0.5f * (c0+c1);
			Vec4f c12 = 0.5f * (c1+c2);
			Vec4f c20 = 0.5f * (c2+c0);

			/*0*/ verticesTessVec.push_back(v0);
			/*1*/ verticesTessVec.push_back(v01);
			/*2*/ verticesTessVec.push_back(v1);
			/*3*/ verticesTessVec.push_back(v12);
			/*4*/ verticesTessVec.push_back(v2);
			/*5*/ verticesTessVec.push_back(v20);

			/*0*/ colorsTessVec.push_back(c0);
			/*1*/ colorsTessVec.push_back(c01);
			/*2*/ colorsTessVec.push_back(c1);
			/*3*/ colorsTessVec.push_back(c12);
			/*4*/ colorsTessVec.push_back(c2);
			/*5*/ colorsTessVec.push_back(c20);

			unsigned int index = 6*i;

			std::vector<unsigned int> f0;
			f0.push_back(index+0);
			f0.push_back(index+1);
			f0.push_back(index+5);
			indicesTessVec.push_back(f0);

			std::vector<unsigned int> f1;
			f1.push_back(index+1);
			f1.push_back(index+2);
			f1.push_back(index+3);
			indicesTessVec.push_back(f1);

			std::vector<unsigned int> f2;
			f2.push_back(index+5);
			f2.push_back(index+3);
			f2.push_back(index+4);
			indicesTessVec.push_back(f2);

			std::vector<unsigned int> f3;
			f3.push_back(index+1);
			f3.push_back(index+3);
			f3.push_back(index+5);
			indicesTessVec.push_back(f3);
		}
	}
	else if(primType == MESHPRIM_TYPE_QUADS)
	{
		for(unsigned int i = 0; i < indicesVec.size(); ++i)
		{
			Vec4f v0 = position0Vec[ indicesVec[i][0] ];
			Vec4f v1 = position0Vec[ indicesVec[i][1] ];
			Vec4f v2 = position0Vec[ indicesVec[i][2] ];
			Vec4f v3 = position0Vec[ indicesVec[i][3] ];

			Vec4f v01 = 0.5f * (v0+v1);
			Vec4f v12 = 0.5f * (v1+v2);
			Vec4f v23 = 0.5f * (v2+v3);
			Vec4f v30 = 0.5f * (v3+v0);

			Vec4f vc = 0.25f * (v0+v1+v2+v3);

			Vec4f c0 = colors0Vec[ indicesVec[i][0] ];
			Vec4f c1 = colors0Vec[ indicesVec[i][1] ];
			Vec4f c2 = colors0Vec[ indicesVec[i][2] ];
			Vec4f c3 = colors0Vec[ indicesVec[i][3] ];

			Vec4f c01 = 0.5f * (c0+c1);
			Vec4f c12 = 0.5f * (c1+c2);
			Vec4f c23 = 0.5f * (c2+c3);
			Vec4f c30 = 0.5f * (c3+c0);

			Vec4f cc = 0.25f * (c0+c1+c2+c3);

			/*0*/ verticesTessVec.push_back(v0);
			/*1*/ verticesTessVec.push_back(v01);
			/*2*/ verticesTessVec.push_back(v1);
			/*3*/ verticesTessVec.push_back(v12);
			/*4*/ verticesTessVec.push_back(v2);
			/*5*/ verticesTessVec.push_back(v23);
			/*6*/ verticesTessVec.push_back(v3);
			/*7*/ verticesTessVec.push_back(v30);

			/*8*/ verticesTessVec.push_back(vc);

			/*0*/ colorsTessVec.push_back(c0);
			/*1*/ colorsTessVec.push_back(c01);
			/*2*/ colorsTessVec.push_back(c1);
			/*3*/ colorsTessVec.push_back(c12);
			/*4*/ colorsTessVec.push_back(c2);
			/*5*/ colorsTessVec.push_back(c23);
			/*6*/ colorsTessVec.push_back(c3);
			/*7*/ colorsTessVec.push_back(c30);

			/*8*/ colorsTessVec.push_back(cc);

			unsigned int index = 9*i;

			std::vector<unsigned int> f0;
			f0.push_back(index+0);
			f0.push_back(index+1);
			f0.push_back(index+8);
			f0.push_back(index+7);
			indicesTessVec.push_back(f0);

			std::vector<unsigned int> f1;
			f1.push_back(index+1);
			f1.push_back(index+2);
			f1.push_back(index+3);
			f1.push_back(index+8);
			indicesTessVec.push_back(f1);

			std::vector<unsigned int> f2;
			f2.push_back(index+8);
			f2.push_back(index+3);
			f2.push_back(index+4);
			f2.push_back(index+5);
			indicesTessVec.push_back(f2);

			std::vector<unsigned int> f3;
			f3.push_back(index+7);
			f3.push_back(index+8);
			f3.push_back(index+5);
			f3.push_back(index+6);
			indicesTessVec.push_back(f3);
		}
	}

	position0Vec = verticesTessVec;
	colors0Vec = colorsTessVec;
	indicesVec = indicesTessVec;

	numVertices = (unsigned int)position0Vec.size();
	numPrimitives = (unsigned int)indicesVec.size();
}

void MeshFile::scale(const float s)
{
	scale(Vec4f(s, s, s, 1.0f));
}

void MeshFile::scale(const Vec4f& scaleVec)
{
	for(unsigned int i = 0; i < position0Vec.size(); ++i)
	{
		position0Vec[i] *= scaleVec;
	}
}

void MeshFile::translate(float x, float y, float z)
{
	translate(Vec4f(x,y,z,0.0f));
}

void MeshFile::translate(const Vec4f& translateVec)
{
	for(unsigned int i = 0; i < position0Vec.size(); ++i)
	{
		position0Vec[i] += translateVec;
	}
}

void MeshFile::scaleToUnitCube()
{
	AABB4f aabb = getAABB();
	Vec4f aabbCenter = center(aabb);
	aabbCenter.w = 0.0f;

	for(unsigned int i = 0; i < position0Vec.size(); ++i)
	{
		position0Vec[i] -= aabbCenter;
	}

	Vec4f aabbSize = size(aabb);
	int max = maxDim( aabbSize );

	// uniform scale
	float scale = 2.0f / aabbSize[max];
	for(unsigned int i = 0; i < position0Vec.size(); ++i)
	{
		position0Vec[i] *= Vec4f(scale, scale, scale, 1.0f);
	}
}

void MeshFile::flattenIndices()
{
	indicesVecFlat.clear();

	if(primType == MESHPRIM_TYPE_TRIS)
	{
		for(unsigned int i = 0; i < indicesVec.size(); ++i)
		{
			indicesVecFlat.push_back( indicesVec[i][0] );
			indicesVecFlat.push_back( indicesVec[i][1] );
			indicesVecFlat.push_back( indicesVec[i][2] );
		}
	}
	else if(primType == MESHPRIM_TYPE_QUADS)
	{
		for(unsigned int i = 0; i < indicesVec.size(); ++i)
		{
			indicesVecFlat.push_back( indicesVec[i][0] );
			indicesVecFlat.push_back( indicesVec[i][1] );
			indicesVecFlat.push_back( indicesVec[i][2] );
			indicesVecFlat.push_back( indicesVec[i][3] );
		}
	}
	else if(primType == MESHPRIM_TYPE_LINES)
	{
		for(unsigned int i = 0; i < indicesVec.size(); ++i)
		{
			indicesVecFlat.push_back( indicesVec[i][0] );
			indicesVecFlat.push_back( indicesVec[i][1] );
		}
	}
}

void MeshFile::saveRaw() const
{
	if(position0Vec.size())
	{
		std::stringstream currFileName;
		currFileName << coreName << "_" << attribName[0] << ".dat";

		std::ofstream myfile(currFileName.str().c_str(), std::ios::binary);
		myfile.write((char*)&position0Vec[0], position0Vec.size() * sizeof(Vec4f));
		myfile.close();
	}

	if(position1Vec.size())
	{
		std::stringstream currFileName;
		currFileName << coreName << "_" << attribName[1] << ".dat";

		std::ofstream myfile(currFileName.str().c_str(), std::ios::binary);
		myfile.write((char*)&position1Vec[0], position1Vec.size() * sizeof(Vec4f));
		myfile.close();
	}

	if(normals0Vec.size())
	{
		std::stringstream currFileName;
		currFileName << coreName << "_" << attribName[2] << ".dat";

		std::ofstream myfile(currFileName.str().c_str(), std::ios::binary);
		myfile.write((char*)&normals0Vec[0], normals0Vec.size() * sizeof(Vec4f));
		myfile.close();
	}

	if(normals1Vec.size())
	{
		std::stringstream currFileName;
		currFileName << coreName << "_" << attribName[3] << ".dat";

		std::ofstream myfile(currFileName.str().c_str(), std::ios::binary);
		myfile.write((char*)&normals1Vec[0], normals1Vec.size() * sizeof(Vec4f));
		myfile.close();
	}

	if(colors0Vec.size())
	{
		std::stringstream currFileName;
		currFileName << coreName << "_" << attribName[4] << ".dat";

		std::ofstream myfile(currFileName.str().c_str(), std::ios::binary);
		myfile.write((char*)&colors0Vec[0], colors0Vec.size() * sizeof(Vec4f));
		myfile.close();
	}

	if(colors1Vec.size())
	{
		std::stringstream currFileName;
		currFileName << coreName << "_" << attribName[5] << ".dat";

		std::ofstream myfile(currFileName.str().c_str(), std::ios::binary);
		myfile.write((char*)&colors1Vec[0], colors1Vec.size() * sizeof(Vec4f));
		myfile.close();
	}

	{
		std::stringstream currFileName;
		currFileName << coreName << "_" << "index" << ".dat";

		std::ofstream myfile(currFileName.str().c_str(), std::ios::binary);
		myfile.write((char*)&indicesVecFlat[0], indicesVecFlat.size() * sizeof(unsigned int));
		myfile.close();
	}
}

bool MeshFile::loadRaw()
{
	return false;
}

void MeshFile::reset()
{
	attribName[0] = "vertices0";
	attribName[1] = "vertices1";
	attribName[2] = "normals0";
	attribName[3] = "normals1";
	attribName[4] = "colors0";
	attribName[5] = "colors1";
	attribName[6] = "uvcoord0";
	attribName[7] = "uvcoord1";

	hasAttribute[0] = false;
	hasAttribute[1] = false;
	hasAttribute[2] = false;
	hasAttribute[3] = false;
	hasAttribute[4] = false;
	hasAttribute[5] = false;
	hasAttribute[6] = false;
	hasAttribute[7] = false;

	position0Vec.clear();
	position1Vec.clear();
	normals0Vec.clear();
	normals1Vec.clear();
	colors0Vec.clear();
	colors1Vec.clear();
	uvcoord0Vec.clear();
	uvcoord1Vec.clear();

	boneindicesVec.clear();
	boneweightsVec.clear();

	indicesVec.clear();
	indicesVecFlat.clear();
	attributeVec.clear();
	usemtlVec.clear();

	primType = MESHPRIM_TYPE_NOTSET;
	numVertsPrim = 0;

	numPrimitives = numVertices = 0;
}