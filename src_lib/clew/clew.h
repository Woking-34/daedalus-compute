#ifndef CLEW_H
#define CLEW_H

#ifdef __APPLE__
	#include <OpenCL/cl.h>
	#include <OpenCL/cl_gl.h>
	#include <OpenCL/cl_ext.h>
#else
	#include <CL/cl.h>
	#include <CL/cl_gl.h>
	#include <CL/cl_ext.h>
#endif

extern cl_int (CL_API_CALL *__GetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
extern cl_int (CL_API_CALL *__GetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
extern cl_int (CL_API_CALL *__GetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
extern cl_int (CL_API_CALL *__GetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *, size_t *);

extern cl_context (CL_API_CALL *__CreateContext)(cl_context_properties *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
extern cl_int (CL_API_CALL *__RetainContext)(cl_context);
extern cl_int (CL_API_CALL *__ReleaseContext)(cl_context);

extern cl_command_queue (CL_API_CALL *__CreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
extern cl_int (CL_API_CALL *__RetainCommandQueue)(cl_command_queue);
extern cl_int (CL_API_CALL *__ReleaseCommandQueue)(cl_command_queue);

extern cl_mem (CL_API_CALL *__CreateBuffer)(cl_context,	cl_mem_flags, size_t, void *, cl_int *);
extern cl_mem (CL_API_CALL *__CreateImage2D)(cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, void *, cl_int *);
extern cl_mem (CL_API_CALL *__CreateImage3D)(cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, size_t, size_t, void *, cl_int *);
extern cl_int (CL_API_CALL *__GetSupportedImageFormats)(cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, cl_image_format *, cl_uint *);
extern cl_int (CL_API_CALL *__RetainMemObject)(cl_mem);
extern cl_int (CL_API_CALL *__ReleaseMemObject)(cl_mem);

extern cl_mem (CL_API_CALL *__CreateFromGLBuffer)(cl_context, cl_mem_flags, cl_GLuint, cl_int *);
extern cl_mem (CL_API_CALL *__CreateFromGLTexture2D)(cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int *);
extern cl_mem (CL_API_CALL *__CreateFromGLTexture3D)(cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int *);

extern cl_int (CL_API_CALL *__EnqueueAcquireGLObjects)(cl_command_queue, cl_uint, const cl_mem *, cl_uint, const cl_event *, cl_event *);
extern cl_int (CL_API_CALL *__EnqueueReleaseGLObjects)(cl_command_queue, cl_uint, const cl_mem *, cl_uint, const cl_event *, cl_event *);

extern cl_program (CL_API_CALL *__CreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
extern cl_program (CL_API_CALL *__CreateProgramWithBinary)(cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *);
extern cl_int (CL_API_CALL *__RetainProgram)(cl_program);
extern cl_int (CL_API_CALL *__ReleaseProgram)(cl_program);

extern cl_int (CL_API_CALL *__BuildProgram)(cl_program,	cl_uint, const cl_device_id *, const char *, void *, void *);
extern cl_int (CL_API_CALL *__GetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);

extern cl_kernel (CL_API_CALL *__CreateKernel)(cl_program, const char *, cl_int *);
extern cl_int (CL_API_CALL *__CreateKernelsInProgram)(cl_program, cl_uint,	cl_kernel *, cl_uint *);
extern cl_int (CL_API_CALL *__RetainKernel)(cl_kernel);
extern cl_int (CL_API_CALL *__ReleaseKernel)(cl_kernel);

extern cl_int (CL_API_CALL *__SetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
extern cl_int (CL_API_CALL *__GetKernelInfo)(cl_kernel, cl_kernel_info, size_t, void *, size_t *);
extern cl_int (CL_API_CALL *__GetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *);

extern cl_int (CL_API_CALL *__Flush)(cl_command_queue);
extern cl_int (CL_API_CALL *__Finish)(cl_command_queue);

extern cl_int (CL_API_CALL *__EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, 	cl_event *);
extern cl_int (CL_API_CALL *__EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
extern cl_int (CL_API_CALL *__EnqueueCopyBuffer)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);

extern cl_int (CL_API_CALL *__EnqueueReadImage)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
extern cl_int (CL_API_CALL *__EnqueueWriteImage)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
extern cl_int (CL_API_CALL *__EnqueueCopyImage)(cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);

extern cl_int (CL_API_CALL *__EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);

extern cl_int (CL_API_CALL *__GetGLContextInfoKHR)(const cl_context_properties*, cl_gl_context_info, size_t, void*, size_t*);

extern void* (CL_API_CALL *__GetExtensionFunctionAddress)(const char *);
extern void* (CL_API_CALL *__GetExtensionFunctionAddressForPlatform)(cl_platform_id, const char *);

#define	clGetPlatformIDs							__GetPlatformIDs
#define	clGetPlatformInfo							__GetPlatformInfo
#define	clGetDeviceIDs								__GetDeviceIDs
#define	clGetDeviceInfo								__GetDeviceInfo
#define	clCreateContext								__CreateContext
#define	clCreateContextFromType						__CreateContextFromType
#define	clRetainContext								__RetainContext
#define	clReleaseContext							__ReleaseContext
#define	clGetContextInfo							__GetContextInfo
#define	clCreateCommandQueue						__CreateCommandQueue
#define	clRetainCommandQueue						__RetainCommandQueue
#define	clReleaseCommandQueue						__ReleaseCommandQueue
#define	clGetCommandQueueInfo						__GetCommandQueueInfo
#define	clCreateBuffer								__CreateBuffer
#define	clCreateSubBuffer							__CreateSubBuffer
#define	clCreateImage2D								__CreateImage2D
#define	clCreateImage3D								__CreateImage3D
#define	clGetSupportedImageFormats					__GetSupportedImageFormats
#define	clRetainMemObject							__RetainMemObject
#define	clReleaseMemObject							__ReleaseMemObject
#define	clCreateFromGLBuffer						__CreateFromGLBuffer
#define	clCreateFromGLTexture2D						__CreateFromGLTexture2D
#define	clCreateFromGLTexture3D						__CreateFromGLTexture3D
#define clEnqueueAcquireGLObjects					__EnqueueAcquireGLObjects
#define clEnqueueReleaseGLObjects					__EnqueueReleaseGLObjects
#define	clGetMemObjectInfo							__GetMemObjectInfo
#define	clGetImageInfo								__GetImageInfo
#define	clCreateSampler								__CreateSampler
#define	clRetainSampler								__RetainSampler
#define	clReleaseSampler							__ReleaseSampler
#define	clGetSamplerInfo							__GetSamplerInfo
#define	clCreateProgramWithSource					__CreateProgramWithSource
#define	clCreateProgramWithBinary					__CreateProgramWithBinary
#define	clRetainProgram								__RetainProgram
#define	clReleaseProgram							__ReleaseProgram
#define	clBuildProgram								__BuildProgram
#define	clUnloadCompiler							__UnloadCompiler
#define	clGetProgramInfo							__GetProgramInfo
#define	clGetProgramBuildInfo						__GetProgramBuildInfo
#define	clCreateKernel								__CreateKernel
#define	clCreateKernelsInProgram					__CreateKernelsInProgram
#define	clRetainKernel								__RetainKernel
#define	clReleaseKernel								__ReleaseKernel
#define	clSetKernelArg								__SetKernelArg
#define	clGetKernelInfo								__GetKernelInfo
#define	clGetKernelWorkGroupInfo					__GetKernelWorkGroupInfo
#define	clWaitForEvents								__WaitForEvents
#define	clGetEventInfo								__GetEventInfo
#define	clCreateUserEvent							__CreateUserEvent
#define	clRetainEvent								__RetainEvent
#define	clReleaseEvent								__ReleaseEvent
#define	clSetUserEventStatus						__SetUserEventStatus
#define	clSetEventCallback							__SetEventCallback
#define	clGetEventProfilingInfo						__GetEventProfilingInfo
#define	clFlush										__Flush
#define	clFinish									__Finish
#define	clEnqueueReadBuffer							__EnqueueReadBuffer
#define	clEnqueueReadBufferRect						__EnqueueReadBufferRect
#define	clEnqueueWriteBuffer						__EnqueueWriteBuffer
#define	clEnqueueWriteBufferRect					__EnqueueWriteBufferRect
#define	clEnqueueCopyBuffer							__EnqueueCopyBuffer
#define	clEnqueueCopyBufferRect						__EnqueueCopyBufferRect
#define	clEnqueueReadImage							__EnqueueReadImage
#define	clEnqueueWriteImage							__EnqueueWriteImage
#define	clEnqueueCopyImage							__EnqueueCopyImage
#define	clEnqueueCopyImageToBuffer					__EnqueueCopyImageToBuffer
#define	clEnqueueCopyBufferToImage					__EnqueueCopyBufferToImage
#define	clEnqueueMapBuffer							__EnqueueMapBuffer
#define	clEnqueueMapImage							__EnqueueMapImage
#define	clEnqueueUnmapMemObject						__EnqueueUnmapMemObject
#define	clEnqueueNDRangeKernel						__EnqueueNDRangeKernel
#define	clEnqueueTask								__EnqueueTask
#define	clEnqueueNativeKernel						__EnqueueNativeKernel
#define	clEnqueueMarker								__EnqueueMarker
#define	clEnqueueWaitForEvents						__EnqueueWaitForEvents
#define	clEnqueueBarrier							__EnqueueBarrier
#define clGetGLContextInfoKHR						__GetGLContextInfoKHR
#define	clGetExtensionFunctionAddress				__GetExtensionFunctionAddress
#define	clGetExtensionFunctionAddressForPlatform	__GetExtensionFunctionAddressForPlatform

int initClew(char* oclPath = NULL);

int releaseClew();

#endif