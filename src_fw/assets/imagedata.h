#ifndef IMAGEDATA_H
#define IMAGEDATA_H

#include "system/platform.h"
#include "system/filesystem.h"
#include "system/log.h"

//enum ImageType
//{
//	IMAGE_TYPE_UCHAR,
//	IMAGE_TYPE_UCHAR2,
//	IMAGE_TYPE_UCHAR4,
//
//	IMAGE_TYPE_FLOAT,
//	IMAGE_TYPE_FLOAT3,
//	IMAGE_TYPE_FLOAT4,
//};

class ImageData
{
public:
	ImageData()
	{
		width = height = numChannels = 0;
		data = nullptr;
		dataFLT = nullptr;
	}

	~ImageData()
	{
		clear();
	}

	int getWidth() const
	{
		return width;
	}

	int getHeight() const
	{
		return height;
	}

	int getNumChannels() const
	{
		return numChannels;
	}

	unsigned char* getData()
	{
		return data;
	}

	float* getDataFLT()
	{
		return dataFLT;
	}

	bool loadPBM(const std::string& fileName);
	
	bool loadBMP(const std::string& fileName);

	void toRGBA();

protected:
	int width;
	int height;
	int numChannels;

	unsigned char* data;	// default [0, 255]
	float* dataFLT;			// default [0.0f, 1.0f]

	void clear();
	
	void loadBinary(std::istream& is);

	void loadASCII(std::istream& is);

private:
	ImageData(const ImageData& that);
	ImageData& operator=(const ImageData& that);
};

#endif