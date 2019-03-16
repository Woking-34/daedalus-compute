#include "platform.h"

void savePPM(const std::string& name, unsigned char* src, int width, int height, int numChannels)
{
	std::string ext;
	std::string format;

	{
		if (numChannels == 1)
		{
			format = "P5\n";
			ext = ".pgm";
		}
		else if (numChannels == 3 || numChannels == 4)
		{
			format = "P6\n";
			ext = ".ppm";
		}
		else
		{
			assert(0);
		}

		std::fstream fh((name + ext).c_str(), std::fstream::out | std::fstream::binary);

		fh << format;

		fh << width << "\n" << height << "\n" << 0xff << std::endl;

		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
			{
				if (numChannels == 1)
				{
					fh << (uchar)(src[numChannels * (i + j*width) + 0]);
				}
				else if (numChannels == 3 || numChannels == 4)
				{
					fh << (uchar)(src[numChannels * (i + j*width) + 0]);
					fh << (uchar)(src[numChannels * (i + j*width) + 1]);
					fh << (uchar)(src[numChannels * (i + j*width) + 2]);
				}
				else
				{
					assert(0);
				}
			}
		}

		fh.flush();
		fh.close();
	}
}

void savePPM(const std::string& name, float* src, int width, int height, int numChannels)
{
	std::string ext;
	std::string format;

	{
		if (numChannels == 1)
		{
			format = "P5\n";
			ext = ".pgm";
		}
		else if (numChannels == 3 || numChannels == 4)
		{
			format = "P6\n";
			ext = ".ppm";
		}
		else
		{
			assert(0);
		}

		std::fstream fh((name + ext).c_str(), std::fstream::out | std::fstream::binary);

		fh << format;

		fh << width << "\n" << height << "\n" << 0xff << std::endl;

		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
			{
				if (numChannels == 1)
				{
					fh << (uchar)(src[numChannels * (i + j*width) + 0] * 255.0f + 0.5f);
				}
				else if (numChannels == 3 || numChannels == 4)
				{
					fh << (uchar)(src[numChannels * (i + j*width) + 0] * 255.0f + 0.5f);
					fh << (uchar)(src[numChannels * (i + j*width) + 1] * 255.0f + 0.5f);
					fh << (uchar)(src[numChannels * (i + j*width) + 2] * 255.0f + 0.5f);
				}
				else
				{
					assert(0);
				}
			}
		}

		fh.flush();
		fh.close();
	}
}

void saveOBJ_pos(const std::string& name, unsigned int numVertices, unsigned int numTriangles, float* positionVec, unsigned int positionStride, unsigned int* indices)
{
	std::fstream fh((name + ".obj").c_str(), std::fstream::out);

	for (unsigned int i = 0; i < numVertices; ++i)
	{
		fh << "v " << positionVec[positionStride*i + 0]
			<< " " << positionVec[positionStride*i + 1]
			<< " " << positionVec[positionStride*i + 2]
			<< std::endl;
	}

	if(indices)
	{
		for (unsigned int i = 0; i < numTriangles; ++i)
		{
			fh << "f " << indices[3*i + 0] + 1
				<< " " << indices[3*i + 1] + 1
				<< " " << indices[3*i + 2] + 1
				<< std::endl;
		}
	}
	else
	{
		for (unsigned int i = 0; i < numTriangles; ++i)
		{
			fh << "f " << 3*i + 0 + 1
				<< " " << 3*i + 1 + 1
				<< " " << 3*i + 2 + 1
				<< std::endl;
		}
	}

	fh.flush();
	fh.close();
}

void saveOBJ_pos_norm(const std::string& name, unsigned int numVertices, unsigned int numTriangles, float* positionVec, float* normalVec, unsigned int positionStride, unsigned int normalStride, unsigned int* indices)
{
	std::fstream fh((name + ".obj").c_str(), std::fstream::out);

	for (unsigned int i = 0; i < numVertices; ++i)
	{
		fh << "v " << positionVec[positionStride*i + 0]
			<< " " << positionVec[positionStride*i + 1]
			<< " " << positionVec[positionStride*i + 2]
			<< std::endl;
	}

	for (unsigned int i = 0; i < numVertices; ++i)
	{
		fh << "vn " << normalVec[normalStride*i + 0]
			 << " " << normalVec[normalStride*i + 1]
			 << " " << normalVec[normalStride*i + 2]
			 << std::endl;
	}

	if(indices)
	{
		for (unsigned int i = 0; i < numTriangles; ++i)
		{
			fh << "f " << indices[3*i + 0] + 1 << "//" << indices[3*i + 0] + 1
				<< " " << indices[3*i + 1] + 1 << "//" << indices[3*i + 1] + 1
				<< " " << indices[3*i + 2] + 1 << "//" << indices[3*i + 2] + 1
				<< std::endl;
		}
	}
	else
	{
		for (unsigned int i = 0; i < numTriangles; ++i)
		{
			fh << "f " << 3*i + 0 + 1 << "//" << 3*i + 0 + 1
				<< " " << 3*i + 1 + 1 << "//" << 3*i + 1 + 1
				<< " " << 3*i + 2 + 1 << "//" << 3*i + 2 + 1
				<< std::endl;
		}
	}
}

void saveOBJ_pos_norm(const std::string& name, unsigned int numVertices, unsigned int numTriangles, float* positionnormalVec, unsigned int positionStride, unsigned int normalStride, unsigned int* indices)
{
	std::fstream fh((name + ".obj").c_str(), std::fstream::out);

	for (unsigned int i = 0; i < numVertices; ++i)
	{
		fh << "v " << positionnormalVec[(positionStride + normalStride)*i + 0]
			<< " " << positionnormalVec[(positionStride + normalStride)*i + 1]
			<< " " << positionnormalVec[(positionStride + normalStride)*i + 2]
			<< std::endl;
	}

	for (unsigned int i = 0; i < numVertices; ++i)
	{
		fh << "vn " << positionnormalVec[(positionStride + normalStride)*i + positionStride + 0]
			 << " " << positionnormalVec[(positionStride + normalStride)*i + positionStride + 1]
			 << " " << positionnormalVec[(positionStride + normalStride)*i + positionStride + 2]
			 << std::endl;
	}

	if(indices)
	{
		for (unsigned int i = 0; i < numTriangles; ++i)
		{
			fh << "f " << indices[3*i + 0] + 1 << "//" << indices[3*i + 0] + 1
				<< " " << indices[3*i + 1] + 1 << "//" << indices[3*i + 1] + 1
				<< " " << indices[3*i + 2] + 1 << "//" << indices[3*i + 2] + 1
				<< std::endl;
		}
	}
	else
	{
		for (unsigned int i = 0; i < numTriangles; ++i)
		{
			fh << "f " << 3*i + 0 + 1 << "//" << 3*i + 0 + 1
				<< " " << 3*i + 1 + 1 << "//" << 3*i + 1 + 1
				<< " " << 3*i + 2 + 1 << "//" << 3*i + 2 + 1
				<< std::endl;
		}
	}
}

uchar* loadRaw(const std::string& fileName, size_t* fileSize)
{
	uchar* rawPtr = NULL;
	size_t _fileSize = 0;

	std::ifstream myfile(fileName.c_str(), std::ios::binary);
	if( myfile.is_open() )
	{
		std::streampos begin = myfile.tellg();
		myfile.seekg(0, std::ios::end);
		std::streampos end = myfile.tellg();
		myfile.seekg(0, std::ios::beg);

		_fileSize = (size_t)(end - begin);
		rawPtr = new uchar[_fileSize];

		myfile.read((char*)rawPtr, _fileSize);
	}

	if( fileSize )
	{
		*fileSize = _fileSize;
	}

	return rawPtr;
}

void saveRaw(const std::string& fileName, size_t fileSize, void* data)
{
	std::ofstream myfile(fileName.c_str(), std::ios::binary);
	myfile.write((char*)data, fileSize);
}

std::string loadStr(const std::string fileName)
{
	std::ifstream myfile(fileName.c_str());
	std::string fileStr( std::istreambuf_iterator<char>(myfile), (std::istreambuf_iterator<char>()) );

	return fileStr;
}

std::string tolowerString(const std::string& str)
{
	std::string ret = str;
	std::transform(ret.begin(), ret.end(), ret.begin(), ::tolower);

	return ret;
}

std::string trimString(const std::string& str, const std::string& whitespace)
{
	size_t strBegin = str.find_first_not_of(whitespace);
	if (strBegin == std::string::npos)
		return "";

	size_t strEnd = str.find_last_not_of(whitespace);
	size_t strRange = strEnd - strBegin + 1;

	return str.substr(strBegin, strRange);
}

std::string reduceString(const std::string& str, const std::string& fill, const std::string& whitespace)
{
	std::string result = trimString(str, whitespace);

	size_t beginSpace = result.find_first_of(whitespace);
	while (beginSpace != std::string::npos)
	{
		size_t endSpace = result.find_first_not_of(whitespace, beginSpace);
		size_t range = endSpace - beginSpace;

		result.replace(beginSpace, range, fill);

		size_t newStart = beginSpace + fill.length();
		beginSpace = result.find_first_of(whitespace, newStart);
	}

	return result;
}

std::string formatInt(int i, int width, char pad)
{
	std::stringstream ss;
	ss << std::setfill(pad) << std::setw(width) << std::fixed << i;

	return ss.str();
}

std::string formatFloat(float f, int width, int precision, char pad)
{
	std::stringstream ss;
	ss << std::setprecision(precision) << std::setfill(pad) << std::setw(width) << std::fixed << f;

	return ss.str();
}

std::string formatDouble(double d, int width, int precision, char pad)
{
	std::stringstream ss;
	ss << std::setprecision(precision) << std::setfill(pad) << std::setw(width) << std::fixed << d;

	return ss.str();
}