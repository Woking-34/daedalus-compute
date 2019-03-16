#include "imagedata.h"

bool ImageData::loadPBM(const std::string& fileName)
{
	clear();

	std::ifstream inPBM;
	inPBM.open( (FileSystem::GetImagesFolder() + fileName).c_str(), std::ios::binary );

	bool isFoundSource = inPBM.is_open();
	
	LOG_BOOL( isFoundSource, LogLine() << "TEXDATA FILEOPEN" << " - " << fileName );

	if( isFoundSource == false )
		return false;

	//	formatStr table - wikipedia
	//	P1 	Portable bitmap 	ASCII
	//	P2 	Portable graymap 	ASCII
	//	P3 	Portable pixmap 	ASCII
	//	P4 	Portable bitmap 	Binary
	//	P5 	Portable graymap 	Binary
	//	P6 	Portable pixmap 	Binary

	std::string formatStr;
	getline(inPBM, formatStr);
	std::stringstream formatSS(formatStr);
	formatSS >> formatStr;

	std::string commentStr;
	while( inPBM.peek() == '#' )
		getline(inPBM, commentStr);

	std::string resolutionStr;
	getline(inPBM, resolutionStr);
	std::stringstream resolutionSS(resolutionStr);
	resolutionSS >> width;
	resolutionSS >> height;

	size_t maxvalue;

	std::string maxvalueStr;
	getline(inPBM, maxvalueStr);
	std::stringstream maxvalueSS(maxvalueStr);
	maxvalueSS >> maxvalue;

	if(width <= 0 || height <= 0)
		return false;

	if(formatStr == "P2" || formatStr == "P5")
	{
		numChannels = 1;
	}
	else if(formatStr == "P3" || formatStr == "P6")
	{
		numChannels = 3;
	}

	data = new unsigned char[numChannels*width*height];
	dataFLT = new float[numChannels*width*height];

	if(formatStr == "P2" || formatStr == "P3")
	{
		loadASCII(inPBM);
	}
	else if(formatStr == "P5" || formatStr == "P6")
	{
		loadBinary(inPBM);
	}

	if(numChannels == 1)
	{
		for(int i = 0; i < width*height; ++i)
		{
			dataFLT[i] = data[i] / 255.0f;
		}
	}
	else if(numChannels == 3)
	{
		for(int i = 0; i < width*height; ++i)
		{
			dataFLT[i*3+0] = data[i*3+0] / 255.0f;
			dataFLT[i*3+1] = data[i*3+1] / 255.0f;
			dataFLT[i*3+2] = data[i*3+2] / 255.0f;
		}
	}
	else
	{
		assert(0);
	}

	inPBM.close();

	LOG_OK( LogLine() << "TEXDATA W x H" << " - " << width << " x " << height );
	LOG("");

	return true;
}

bool ImageData::loadBMP(const std::string& fileName)
{
	clear();

	std::ifstream inBMP;
	inBMP.open( (FileSystem::GetImagesFolder() + fileName).c_str(), std::ios::binary );

	bool isFoundSource = inBMP.is_open();
	
	LOG_BOOL( isFoundSource, LogLine() << "TEXDATA FILEOPEN" << " - " << fileName );

	if( isFoundSource == false )
		return false;
	
	//BITMAPFILEHEADER bmpheader;
	//BITMAPINFOHEADER bmpinfo;

	size_t bmpheaderSize = 14;
	size_t bmpinfoSize = 40;

	size_t widthOffset = 18;
	size_t heightOffset = 22;

	inBMP.seekg(widthOffset, std::ios::beg);
	width = inBMP.get();
	width += inBMP.get()*256;
	width += inBMP.get()*256*256;
	width += inBMP.get()*256*256*256;
	width = abs( width );

	inBMP.seekg(heightOffset, std::ios::beg);
	height = inBMP.get();
	height += inBMP.get()*256;
	height += inBMP.get()*256*256;
	height += inBMP.get()*256*256*256;
	height = abs( height );

	numChannels = 3;

	data = new unsigned char[numChannels*width*height];

	inBMP.seekg(bmpheaderSize+bmpinfoSize, std::ios::beg);
	loadBinary(inBMP);

	inBMP.close();

	// BGR --> RGB
	for(int i = 0; i < width*height; ++i)
	{
		uchar u0 = data[3*i+0];
		uchar u2 = data[3*i+2];

		data[3*i+2] = u0;
		data[3*i+0] = u2;
	}

	LOG_OK( LogLine() << "TEXDATA W x H" << " - " << width << " x " << height );
	LOG("");

	return true;
}

void ImageData::toRGBA()
{
	unsigned char* rgbaBuff = new unsigned char[4*width*height];
	float* rgbaBuffFLT = new float[4*width*height];

	if(numChannels == 1)
	{
		for(int i = 0; i < width*height; ++i)
		{
			rgbaBuff[i*4+0] = data[i];
			rgbaBuff[i*4+1] = data[i];
			rgbaBuff[i*4+2] = data[i];
			rgbaBuff[i*4+3] = 255;

			rgbaBuffFLT[i*4+0] = dataFLT[i];
			rgbaBuffFLT[i*4+1] = dataFLT[i];
			rgbaBuffFLT[i*4+2] = dataFLT[i];
			rgbaBuffFLT[i*4+3] = 1.0f;
		}
	}
	else if(numChannels == 3)
	{
		for(int i = 0; i < width*height; ++i)
		{
			rgbaBuff[i*4+0] = data[i*3+0];
			rgbaBuff[i*4+1] = data[i*3+1];
			rgbaBuff[i*4+2] = data[i*3+2];
			rgbaBuff[i*4+3] = 255;

			rgbaBuffFLT[i*4+0] = dataFLT[i*3+0];
			rgbaBuffFLT[i*4+1] = dataFLT[i*3+1];
			rgbaBuffFLT[i*4+2] = dataFLT[i*3+2];
			rgbaBuffFLT[i*4+3] = 1.0f;
		}
	}
	else
	{
		assert(0);
	}

	delete[] data;
	data = rgbaBuff;

	delete[] dataFLT;
	dataFLT = rgbaBuffFLT;

	numChannels = 4;
}

void ImageData::clear()
{
	DEALLOC_ARR(data);
	DEALLOC_ARR(dataFLT);

	width = height = numChannels = 0;
}

void ImageData::loadBinary(std::istream& is)
{
	is.read((char*)data, numChannels*width*height*sizeof(unsigned char));
}

void ImageData::loadASCII(std::istream& is)
{
	std::istreambuf_iterator<char> eos;
	std::string fileStr(std::istreambuf_iterator<char>(is), eos);
	std::stringstream fileSS(fileStr);

	for (int j = 0; j < height; ++j)
	{
		for (int i = 0; i < width; ++i)
		{
			if(numChannels == 1)
			{
				int grey = 0;

				fileSS >> grey;

				data[i + j*width] = grey;
			}
			else
			{
				int r = 0;
				int g = 0;
				int b = 0;

				fileSS >> r;
				fileSS >> g;
				fileSS >> b;

				data[3 * (i + j*width) + 0] = r;
				data[3 * (i + j*width) + 1] = g;
				data[3 * (i + j*width) + 2] = b;
			}
		}
	}

	/*
	for(int j = 0; j < height; ++j)
	{
		std::string lineStr;
		getline(is, lineStr);
		std::stringstream lineSS(lineStr);

		for(int i = 0; i < width; ++i)
		{
			if(numChannels == 1)
			{
				int grey = 0;

				lineSS >> grey;

				data[i+j*width] = grey;
			}
			else
			{
				int r = 0;
				int g = 0;
				int b = 0;

				lineSS >> r;
				lineSS >> g;
				lineSS >> b;

				data[3*(i+j*width)+0] = r;
				data[3*(i+j*width)+1] = g;
				data[3*(i+j*width)+2] = b;
			}
		}
	}
	*/
}