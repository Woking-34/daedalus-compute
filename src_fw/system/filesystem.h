#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include "system/platform.h"

class FileSystem
{
public:
	FileSystem() {}
	~FileSystem() {}

	static std::string GetRootFolder();
	static std::string GetMeshesFolder();
	static std::string GetImagesFolder();

	static std::string GetShadersGLFolder();
	static std::string GetShadersDXFolder();

	static std::string GetKernelsGLFolder();
	static std::string GetKernelsCLFolder();
	static std::string GetKernelsCUFolder();

	static std::string GetScriptsFolder();

	static std::string GetRawFolder();

	static void Init();
	static bool FileExists( const std::string& file );

private:
	static std::string sRootFolder;
	static std::string sMeshesSubFolder;
	static std::string sImagesSubFolder;

	static std::string sShadersGLSubFolder;
	static std::string sShadersDXSubFolder;

	static std::string sKernelsGLSubFolder;
	static std::string sKernelsCLSubFolder;
	static std::string sKernelsCUSubFolder;

	static std::string sScriptsSubFolder;

	static std::string sRawSubFolder;
};

#endif