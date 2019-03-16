#include "clew.h"

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #define VC_EXTRALEAN
    #include <windows.h>
	#include <stdlib.h>

	#define CALL_CONV _stdcall

    typedef HMODULE				DYNLIB_HANDLE;

    #define DYNLIB_OPEN			LoadLibrary
    #define DYNLIB_CLOSE		FreeLibrary
    #define DYNLIB_IMPORT		GetProcAddress
#else
    #include <dlfcn.h>
    #include <stdlib.h>

	#define CALL_CONV

    typedef void*				DYNLIB_HANDLE;

    #define DYNLIB_OPEN(path)	dlopen(path, RTLD_NOW | RTLD_GLOBAL)
    #define DYNLIB_CLOSE		dlclose
    #define DYNLIB_IMPORT		dlsym
#endif

#include <string>
#include <vector>
#include <iostream>

static DYNLIB_HANDLE module = NULL;

cl_int (CL_API_CALL *__GetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *) = NULL;
cl_int (CL_API_CALL *__GetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *) = NULL;
cl_int (CL_API_CALL *__GetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void *, size_t *) = NULL;
cl_int (CL_API_CALL *__GetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *, size_t *) = NULL;

cl_context (CL_API_CALL *__CreateContext)(cl_context_properties *, cl_uint, const cl_device_id *, void *, void *, cl_int *) = NULL;
cl_int (CL_API_CALL *__RetainContext)(cl_context) = NULL;
cl_int (CL_API_CALL *__ReleaseContext)(cl_context) = NULL;

cl_command_queue (CL_API_CALL *__CreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int *) = NULL;
cl_int (CL_API_CALL *__RetainCommandQueue)(cl_command_queue) = NULL;
cl_int (CL_API_CALL *__ReleaseCommandQueue)(cl_command_queue) = NULL;

cl_mem (CL_API_CALL *__CreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *) = NULL;
cl_mem (CL_API_CALL *__CreateImage2D)(cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, void *, cl_int *) = NULL;
cl_mem (CL_API_CALL *__CreateImage3D)(cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, size_t, size_t, void *, cl_int *) = NULL;
cl_int (CL_API_CALL *__GetSupportedImageFormats)(cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, cl_image_format *, cl_uint *) = NULL;
cl_int (CL_API_CALL *__RetainMemObject)(cl_mem) = NULL;
cl_int (CL_API_CALL *__ReleaseMemObject)(cl_mem) = NULL;

cl_mem (CL_API_CALL *__CreateFromGLBuffer)(cl_context, cl_mem_flags, cl_GLuint, cl_int *) = NULL;
cl_mem (CL_API_CALL *__CreateFromGLTexture2D)(cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int *) = NULL;
cl_mem (CL_API_CALL *__CreateFromGLTexture3D)(cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int *) = NULL;

cl_int (CL_API_CALL *__EnqueueAcquireGLObjects)(cl_command_queue, cl_uint, const cl_mem *, cl_uint, const cl_event *, cl_event *) = NULL;
cl_int (CL_API_CALL *__EnqueueReleaseGLObjects)(cl_command_queue, cl_uint, const cl_mem *, cl_uint, const cl_event *, cl_event *) = NULL;

cl_program (CL_API_CALL *__CreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *) = NULL;
cl_program (CL_API_CALL *__CreateProgramWithBinary)(cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *) = NULL;
cl_int (CL_API_CALL *__RetainProgram)(cl_program) = NULL;
cl_int (CL_API_CALL *__ReleaseProgram)(cl_program) = NULL;

cl_int (CL_API_CALL *__BuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *) = NULL;
cl_int (CL_API_CALL *__GetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *) = NULL;

cl_kernel (CL_API_CALL *__CreateKernel)(cl_program, const char *, cl_int *) = NULL;
cl_int (CL_API_CALL *__CreateKernelsInProgram)(cl_program, cl_uint,	cl_kernel *, cl_uint *) = NULL;
cl_int (CL_API_CALL *__RetainKernel)(cl_kernel) = NULL;
cl_int (CL_API_CALL *__ReleaseKernel)(cl_kernel) = NULL;

cl_int (CL_API_CALL *__SetKernelArg)(cl_kernel, cl_uint, size_t, const void *) = NULL;
cl_int (CL_API_CALL *__GetKernelInfo)(cl_kernel, cl_kernel_info, size_t, void *, size_t *) = NULL;
cl_int (CL_API_CALL *__GetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *) = NULL;

cl_int (CL_API_CALL *__Flush)(cl_command_queue) = NULL;
cl_int (CL_API_CALL *__Finish)(cl_command_queue) = NULL;

cl_int (CL_API_CALL *__EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, 	cl_event *) = NULL;
cl_int (CL_API_CALL *__EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *) = NULL;
cl_int (CL_API_CALL *__EnqueueCopyBuffer)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *) = NULL;

cl_int (CL_API_CALL *__EnqueueReadImage)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *) = NULL;
cl_int (CL_API_CALL *__EnqueueWriteImage)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *) = NULL;
cl_int (CL_API_CALL *__EnqueueCopyImage)(cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *) = NULL;

cl_int (CL_API_CALL *__EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *) = NULL;

cl_int (CL_API_CALL *__GetGLContextInfoKHR)(const cl_context_properties*, cl_gl_context_info, size_t, void*, size_t*) = NULL;

void* (CL_API_CALL *__GetExtensionFunctionAddress)(const char *) = NULL;
void* (CL_API_CALL *__GetExtensionFunctionAddressForPlatform)(cl_platform_id, const char *) = NULL;

int initClew(char* oclPath)
{
	std::vector< std::string > clPathVec;

	if(oclPath == NULL)
	{
#ifdef BUILD_WINDOWS
	clPathVec.push_back("OpenCL.dll");
#endif

#ifdef BUILD_UNIX
	clPathVec.push_back("libOpenCL.so.1");
#endif

#ifdef BUILD_APPLE
	clPathVec.push_back("/System/Library/Frameworks/OpenCL.framework/Versions/Current/OpenCL");
#endif

#ifdef BUILD_ANDROID
	clPathVec.push_back("/system/lib/libOpenCL.so");
	clPathVec.push_back("/vendor/lib/libOpenCL.so");
	clPathVec.push_back("/system/vendor/lib/libOpenCL.so");

	clPathVec.push_back("/vendor/lib/egl/libGLES_mali.so");
	clPathVec.push_back("/system/vendor/lib/egl/libGLES_mali.so");

	clPathVec.push_back("/vendor/lib/libPVROCL.so");
#endif
	}
	else
	{
		clPathVec.push_back(oclPath);
	}

	bool oclFound = false;

	for(unsigned int i = 0; i < clPathVec.size(); ++i)
	{
		module = DYNLIB_OPEN( clPathVec[i].c_str() );

		if(module != NULL)
		{
			*(void **)(&__GetPlatformIDs) = DYNLIB_IMPORT(module, "clGetPlatformIDs");
			*(void **)(&__GetDeviceIDs) = DYNLIB_IMPORT(module, "clGetDeviceIDs");
			*(void **)(&__GetPlatformInfo) = DYNLIB_IMPORT(module, "clGetPlatformInfo");
			*(void **)(&__GetDeviceInfo) = DYNLIB_IMPORT(module, "clGetDeviceInfo");

			if(__GetPlatformIDs && __GetDeviceIDs && __GetPlatformInfo && __GetDeviceInfo)
			{
				std::cout << "OpenCL loaded from: " << clPathVec[i] << std::endl << std::endl;
				oclFound = true;
				break;
			}

			DYNLIB_CLOSE(module);
		}
	}

	if(oclFound == false)
	{
		std::cout << "Could not load OpenCL" << std::endl;
		return 1;
	}
	else
	{
		*(void **)(&__CreateContext) = DYNLIB_IMPORT(module, "clCreateContext");
		*(void **)(&__RetainContext) = DYNLIB_IMPORT(module, "clRetainContext");
		*(void **)(&__ReleaseContext) = DYNLIB_IMPORT(module, "clReleaseContext");

		*(void **)(&__CreateCommandQueue) = DYNLIB_IMPORT(module, "clCreateCommandQueue");
		*(void **)(&__RetainCommandQueue) = DYNLIB_IMPORT(module, "clRetainCommandQueue");
		*(void **)(&__ReleaseCommandQueue) = DYNLIB_IMPORT(module, "clReleaseCommandQueue");

		*(void **)(&__CreateBuffer) = DYNLIB_IMPORT(module, "clCreateBuffer");
		*(void **)(&__CreateImage2D) = DYNLIB_IMPORT(module, "clCreateImage2D");
		*(void **)(&__CreateImage3D) = DYNLIB_IMPORT(module, "clCreateImage3D");
		*(void **)(&__GetSupportedImageFormats) = DYNLIB_IMPORT(module, "clGetSupportedImageFormats");
		*(void **)(&__RetainMemObject) = DYNLIB_IMPORT(module, "clRetainMemObject");
		*(void **)(&__ReleaseMemObject) = DYNLIB_IMPORT(module, "clReleaseMemObject");

		*(void **)(&__CreateFromGLBuffer) = DYNLIB_IMPORT(module, "clCreateFromGLBuffer");
		*(void **)(&__CreateFromGLTexture2D) = DYNLIB_IMPORT(module, "clCreateFromGLTexture2D");
		*(void **)(&__CreateFromGLTexture3D) = DYNLIB_IMPORT(module, "clCreateFromGLTexture3D");

		*(void **)(&__EnqueueAcquireGLObjects) = DYNLIB_IMPORT(module, "clEnqueueAcquireGLObjects");
		*(void **)(&__EnqueueReleaseGLObjects) = DYNLIB_IMPORT(module, "clEnqueueReleaseGLObjects");

		*(void **)(&__CreateProgramWithSource) = DYNLIB_IMPORT(module, "clCreateProgramWithSource");
		*(void **)(&__CreateProgramWithBinary) = DYNLIB_IMPORT(module, "clCreateProgramWithBinary");
		*(void **)(&__RetainProgram) = DYNLIB_IMPORT(module, "clRetainProgram");
		*(void **)(&__ReleaseProgram) = DYNLIB_IMPORT(module, "clReleaseProgram");

		*(void **)(&__BuildProgram) = DYNLIB_IMPORT(module, "clBuildProgram");
		*(void **)(&__GetProgramBuildInfo) = DYNLIB_IMPORT(module, "clGetProgramBuildInfo");

		*(void **)(&__CreateKernel) = DYNLIB_IMPORT(module, "clCreateKernel");
		*(void **)(&__CreateKernelsInProgram) = DYNLIB_IMPORT(module, "clCreateKernelsInProgram");
		*(void **)(&__RetainKernel) = DYNLIB_IMPORT(module, "clRetainKernel");
		*(void **)(&__ReleaseKernel) = DYNLIB_IMPORT(module, "clReleaseKernel");

		*(void **)(&__SetKernelArg) = DYNLIB_IMPORT(module, "clSetKernelArg");
		*(void **)(&__GetKernelInfo) = DYNLIB_IMPORT(module, "clGetKernelInfo");
		*(void **)(&__GetKernelWorkGroupInfo) = DYNLIB_IMPORT(module, "clGetKernelWorkGroupInfo");

		*(void **)(&__Flush) = DYNLIB_IMPORT(module, "clFlush");
		*(void **)(&__Finish) = DYNLIB_IMPORT(module, "clFinish");

		*(void **)(&__EnqueueReadBuffer) = DYNLIB_IMPORT(module, "clEnqueueReadBuffer");
		*(void **)(&__EnqueueWriteBuffer) = DYNLIB_IMPORT(module, "clEnqueueWriteBuffer");
		*(void **)(&__EnqueueCopyBuffer) = DYNLIB_IMPORT(module, "clEnqueueCopyBuffer");

		*(void **)(&__EnqueueReadImage) = DYNLIB_IMPORT(module, "clEnqueueReadImage");
		*(void **)(&__EnqueueWriteImage) = DYNLIB_IMPORT(module, "clEnqueueWriteImage");
		*(void **)(&__EnqueueCopyImage) = DYNLIB_IMPORT(module, "clEnqueueCopyImage");

		*(void **)(&__EnqueueNDRangeKernel) = DYNLIB_IMPORT(module, "clEnqueueNDRangeKernel");

		*(void **)(&__GetGLContextInfoKHR) = DYNLIB_IMPORT(module, "clGetGLContextInfoKHR");

		*(void **)(&__GetExtensionFunctionAddress) = DYNLIB_IMPORT(module, "clGetExtensionFunctionAddress");
		*(void **)(&__GetExtensionFunctionAddressForPlatform) = DYNLIB_IMPORT(module, "clGetExtensionFunctionAddressForPlatform");

		return 0;
	}
}

int releaseClew()
{
	if(module)
	{
		//  ignore errors
		DYNLIB_CLOSE(module);
		module = 0;
	}

	return 0;
}