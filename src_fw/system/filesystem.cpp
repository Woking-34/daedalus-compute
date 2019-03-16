#include "filesystem.h"
#include "log.h"

std::string FileSystem::sRootFolder = "_resources/";
std::string FileSystem::sMeshesSubFolder = "meshes/";
std::string FileSystem::sImagesSubFolder = "images/";
std::string FileSystem::sShadersGLSubFolder = "shaders_gl/";
std::string FileSystem::sShadersDXSubFolder = "shaders_dx/";
std::string FileSystem::sKernelsGLSubFolder = "kernels_gl/";
std::string FileSystem::sKernelsCLSubFolder = "kernels_cl/";
std::string FileSystem::sKernelsCUSubFolder = "kernels_cu/";
std::string FileSystem::sScriptsSubFolder = "scripts/";
std::string FileSystem::sRawSubFolder = "raw/";

void FileSystem::Init()
{
	bool fsSucc = false;
	const std::string fsLoc = "filesystem.loc";

	for(unsigned int i = 0; i < 16; ++i)
	{
		if( FileExists(sRootFolder + fsLoc) )
		{
			fsSucc = true;
			break;
		}

		sRootFolder = "../" + sRootFolder;
	}

	LOG_BOOL(fsSucc, "FileSystem::Init()\n");
}

std::string FileSystem::GetRootFolder()
{
	return( sRootFolder );
}

std::string FileSystem::GetMeshesFolder()
{
	return( sRootFolder + sMeshesSubFolder );
}

std::string FileSystem::GetImagesFolder()
{
	return( sRootFolder + sImagesSubFolder);
}

std::string FileSystem::GetShadersGLFolder()
{
	return( sRootFolder + sShadersGLSubFolder );
}

std::string FileSystem::GetKernelsGLFolder()
{
	return( sRootFolder + sKernelsGLSubFolder );
}

std::string FileSystem::GetKernelsCLFolder()
{
	return( sRootFolder + sKernelsCLSubFolder );
}

std::string FileSystem::GetKernelsCUFolder()
{
	return( sRootFolder + sKernelsCUSubFolder );
}

std::string FileSystem::GetScriptsFolder()
{
	return( sRootFolder + sScriptsSubFolder );
}

std::string FileSystem::GetRawFolder()
{
	return( sRootFolder + sRawSubFolder );
}

bool FileSystem::FileExists( const std::string& fileName )
{
	std::ifstream file( fileName.c_str() );
	return file.is_open();
}