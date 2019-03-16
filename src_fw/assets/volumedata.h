#ifndef VOLUMEDATA_H
#define VOLUMEDATA_H

#include "system/platform.h"
#include "system/filesystem.h"
#include "system/log.h"

//enum VolumeType
//{
//	VOLUME_TYPE_UCHAR,
//	VOLUME_TYPE_UCHAR4,
//
//	VOLUME_TYPE_FLOAT,
//	VOLUME_TYPE_FLOAT4,
//};

template <class T>
class VolumeData
{
public:
	VolumeData() : data(nullptr), dataFLT(nullptr), clear(zero), clearFLT(zero)
	{
		reset();
	}
	~VolumeData()
	{
		reset();
	}

	T* getData()
	{
		return data;
	}
	float* getDataFLT()
	{
		return dataFLT;
	}

	int getSize()
	{
		return width*height*depth;
	}

	int getW()
	{
		return width;
	}
	int getH()
	{
		return height;
	}
	int getD()
	{
		return depth;
	}

	void setName(const std::string& fileName);
	void setSize(int numChannels, int width, int height, int depth);
	void loadFromFile();

	void setClear(T clear)
	{
		this->clear = clear;
	}

	void setClearFLT(float clearFLT)
	{
		this->clearFLT = clearFLT;
	}

	void pad(int widthPadded, int heightPadded, int depthPadded);
	void padRound(int roundTo);

	void half();

	void scaleFloat(float scale);

	//void flipXVolume();
	//void flipYVolume();
	//void flipZVolume();
	//
	//void rotateXVolume();
	//void rotateYVolume();
	//void rotateZVolume();

protected:
	T* data;
	float* dataFLT;

	T clear;
	float clearFLT;

	std::string fileName;
	std::string coreName;
	std::string extName;

	int numChannels;

	int width, height, depth;

	void reset();
};

template <class T>
void VolumeData<T>::reset()
{
	DEALLOC_ARR(data);
	DEALLOC_ARR(dataFLT);

	numChannels = 0;

	width = height = depth = 0;
}

template <class T>
void VolumeData<T>::setName(const std::string& fileName)
{
	this->fileName = fileName;

	this->coreName = fileName.substr(0, fileName.size()-4);
	this->extName = fileName.substr(fileName.length()-3,3);
}

template <class T>
void VolumeData<T>::setSize(int numChannels, int width, int height, int depth)
{
	this->numChannels = numChannels;

	this->width = width;
	this->height = height;
	this->depth = depth;
}

template <class T>
void VolumeData<T>::loadFromFile()
{
	size_t fileSize = 0;
	data = (T*)loadRaw(FileSystem::GetRawFolder() + fileName, &fileSize);

	bool volumeFileExists = data != nullptr;
	LOG_BOOL( volumeFileExists, LogLine() << "VOLUMEDATA FILEOPEN" << " - " << fileName );

	if(fileSize != width*height*depth*sizeof(T))
	{
		LOG_ERR( LogLine() << "VOLUMEDATA SIZE MISMATCH" << " - " << fileName );
	}
	else
	{
		LOG_OK( LogLine() << "VOLUMEDATA" << " - " << width << " x " << height << " x " << depth );
	}
	
	dataFLT = new float[width*height*depth];
	for(int i = 0; i < width*height*depth; ++i)
	{
		dataFLT[i] = data[i];
	}

	LOG("");
}

template <class T>
void VolumeData<T>::pad(int widthPadded, int heightPadded, int depthPadded)
{
	T* dataPadded = new T[widthPadded*heightPadded*depthPadded];
	float* dataPaddedFLT = new float[widthPadded*heightPadded*depthPadded];

	//memset(dataPadded, 0, widthPadded*heightPadded*depthPadded*sizeof(T));
	//memset(dataPaddedFLT, 0, widthPadded*heightPadded*depthPadded*sizeof(float));

	for (int i = 0; i < widthPadded*heightPadded*depthPadded; ++i)
	{
		dataPadded[i] = clear;
		dataPaddedFLT[i] = clearFLT;
	}

	for (int k = 0; k < depth; ++k)
	{
		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
			{
				dataPadded[i + j*widthPadded + k*widthPadded*heightPadded] = data[i + j*width + k*width*height];
			}
		}
	}

	for (int k = 0; k < depth; ++k)
	{
		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
			{
				dataPaddedFLT[i + j*widthPadded + k*widthPadded*heightPadded] = data[i + j*width + k*width*height];
			}
		}
	}

	std::swap(width, widthPadded);
	std::swap(height, heightPadded);
	std::swap(depth, depthPadded);

	std::swap(data, dataPadded);
	std::swap(dataFLT, dataPaddedFLT);

	delete[] dataPadded;
	delete[] dataPaddedFLT;
}

template <class T>
void VolumeData<T>::padRound(int roundTo)
{
	pad(roundUp(width, roundTo), roundUp(height, roundTo), roundUp(depth, roundTo));
}


template <class T>
void VolumeData<T>::half()
{
	width /= 2;
	height /= 2;
	depth /= 2;

	for (int k = 0; k < depth; ++k)
	{
		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
			{
				int from000 = 2 * i + 2 * j*(2 * width) + 2 * k*(2 * width)*(2 * height);
				int from001 = 2 * i + 1 + 2 * j*(2 * width) + 2 * k*(2 * width)*(2 * height);
				int from010 = 2 * i + (2 * j + 1)*(2 * width) + 2 * k*(2 * width)*(2 * height);
				int from011 = 2 * i + 1 + (2 * j + 1)*(2 * width) + 2 * k*(2 * width)*(2 * height);
				int from100 = 2 * i + 2 * j*(2 * width) + (2 * k + 1)*(2 * width)*(2 * height);
				int from101 = 2 * i + 1 + 2 * j*(2 * width) + (2 * k + 1)*(2 * width)*(2 * height);
				int from110 = 2 * i + (2 * j + 1)*(2 * width) + (2 * k + 1)*(2 * width)*(2 * height);
				int from111 = 2 * i + 1 + (2 * j + 1)*(2 * width) + (2 * k + 1)*(2 * width)*(2 * height);

				int to = i + j*width + k*width*height;

				T newval = data[from000];
				newval += data[from001];
				newval += data[from010];
				newval += data[from011];
				newval += data[from100];
				newval += data[from101];
				newval += data[from110];
				newval += data[from111];
				newval /= 8;

				float newvalFLT = dataFLT[from000];
				newvalFLT += dataFLT[from001];
				newvalFLT += dataFLT[from010];
				newvalFLT += dataFLT[from011];
				newvalFLT += dataFLT[from100];
				newvalFLT += dataFLT[from101];
				newvalFLT += dataFLT[from110];
				newvalFLT += dataFLT[from111];
				newvalFLT /= 8.0f;

				data[to] = newval;
				dataFLT[to] = newvalFLT;
			}
		}
	}
}

template <class T>
void VolumeData<T>::scaleFloat(float scale)
{
	for(int i = 0; i < getSize(); ++i)
	{
		dataFLT[i] = dataFLT[i] / scale;
	}
}

/*
template<typename T>
void VolumeData<T>::flipYVolume()
{
	T* temp = new T[width*height*depth];
	float* tempFLT = new float[width*height*depth];

	T* tempPadded = new T[widthPadded*heightPadded*depthPadded];
	float* tempPaddedFLT = new float[widthPadded*heightPadded*depthPadded];

	for(int k = 0; k < depth; ++k)
	{
		for(int j = 0; j < height; ++j)
		{
			for(int i = 0; i < width; ++i)
			{
				int from = i + j*width + k*width*height;
				int to = i + (height-1-j)*width + k*width*height;

				tempPadded[to] = dataPadded[from];
				tempPaddedFLT[to] = dataPaddedFLT[from];
			}
		}
	}

	for(int k = 0; k < depthPadded; ++k)
	{
		for(int j = 0; j < heightPadded; ++j)
		{
			for(int i = 0; i < widthPadded; ++i)
			{
				int from = i + j*widthPadded + k*widthPadded*heightPadded;
				int to = i + (heightPadded-1-j)*widthPadded + k*widthPadded*heightPadded;

				tempPadded[to] = dataPadded[from];
				tempPaddedFLT[to] = dataPaddedFLT[from];
			}
		}
	}

	memcpy(data, temp, width*height*depth*sizeof(T));
	memcpy(dataFLT, tempFLT, width*height*depth*sizeof(float));

	memcpy(dataPadded, tempPadded, widthPadded*heightPadded*depthPadded*sizeof(T));
	memcpy(dataPaddedFLT, tempPaddedFLT, widthPadded*heightPadded*depthPadded*sizeof(float));

	delete[] temp;
	delete[] tempFLT;

	delete[] tempPadded;
	delete[] tempPaddedFLT;
}

template<typename T>
void VolumeData<T>::flipZVolume()
{
	T* temp = new T[width*height*depth];
	float* tempFLT = new float[width*height*depth];

	T* tempPadded = new T[widthPadded*heightPadded*depthPadded];
	float* tempPaddedFLT = new float[widthPadded*heightPadded*depthPadded];

	for(int k = 0; k < depth; ++k)
	{
		for(int j = 0; j < height; ++j)
		{
			for(int i = 0; i < width; ++i)
			{
				int from = i + j*width + k*width*height;
				int to = i + j*width + (depth-1-k)*width*height;

				temp[to] = data[from];
				tempFLT[to] = dataFLT[from];
			}
		}
	}

	for(int k = 0; k < depthPadded; ++k)
	{
		for(int j = 0; j < heightPadded; ++j)
		{
			for(int i = 0; i < widthPadded; ++i)
			{
				int from = i + j*widthPadded + k*widthPadded*heightPadded;
				int to = i + j*widthPadded + (depthPadded-1-k)*widthPadded*heightPadded;

				tempPadded[to] = dataPadded[from];
				tempPaddedFLT[to] = dataPaddedFLT[from];
			}
		}
	}

	memcpy(data, temp, width*height*depth*sizeof(T));
	memcpy(dataFLT, tempFLT, width*height*depth*sizeof(float));

	memcpy(dataPadded, tempPadded, widthPadded*heightPadded*depthPadded*sizeof(T));
	memcpy(dataPaddedFLT, tempPaddedFLT, widthPadded*heightPadded*depthPadded*sizeof(float));

	delete[] temp;
	delete[] tempFLT;

	delete[] tempPadded;
	delete[] tempPaddedFLT;
}

template<typename T>
void VolumeData<T>::rotateXVolume()
{
	T* temp = new T[width*height*depth];
	float* tempFLT = new float[width*height*depth];

	T* tempPadded = new T[widthPadded*heightPadded*depthPadded];
	float* tempPaddedFLT = new float[widthPadded*heightPadded*depthPadded];

	for(int k = 0; k < depth; ++k)
	{
		for(int j = 0; j < height; ++j)
		{
			for(int i = 0; i < width; ++i)
			{
				int from = i + j*width + k*width*height;
				int to = i + (-k+depth-1)*width+ j*width*depth;

				temp[to] = data[from];
				tempFLT[to] = dataFLT[from];
			}
		}
	}

	for(int k = 0; k < depthPadded; ++k)
	{
		for(int j = 0; j < heightPadded; ++j)
		{
			for(int i = 0; i < widthPadded; ++i)
			{
				int from = i + j*widthPadded + k*widthPadded*heightPadded;
				int to = i + (-k+depthPadded-1)*widthPadded + j*widthPadded*depthPadded;

				tempPadded[to] = dataPadded[from];
				tempPaddedFLT[to] = dataPaddedFLT[from];
			}
		}
	}

	int widthTemp = width;
	int heightTemp = height;
	int depthTemp = depth;

	int widthPaddedTemp = widthPadded;
	int heightPaddedTemp = heightPadded;
	int depthPaddedTemp = depthPadded;

	width = widthTemp;
	widthPadded = widthPaddedTemp;

	height = depthTemp;
	heightPadded = depthPaddedTemp;

	depth = heightTemp;
	depthPadded = heightPaddedTemp;

	memcpy(data, temp, width*height*depth*sizeof(T));
	memcpy(dataFLT, tempFLT, width*height*depth*sizeof(float));

	memcpy(dataPadded, tempPadded, widthPadded*heightPadded*depthPadded*sizeof(T));
	memcpy(dataPaddedFLT, tempPaddedFLT, widthPadded*heightPadded*depthPadded*sizeof(float));

	delete[] temp;
	delete[] tempFLT;

	delete[] tempPadded;
	delete[] tempPaddedFLT;
}
*/
#endif