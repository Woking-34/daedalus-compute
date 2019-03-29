#ifndef CLBASE_H
#define CLBASE_H

#include <CLEW/clew.h>
#include "glutil/glbase.h"

#include "system/log.h"

#define CHECK_CL(clStatus) if(clStatus != CL_SUCCESS) { LOG_ERR( LogLine() << "File: " << __FILE__ << " " << "Line: " << __LINE__ << " " << "OpenCL error: " << openclGetErrorString(clStatus) ); }
std::string openclGetErrorString(cl_int clStatus);

// retrieve a string value of platform info
inline void get_platform_info(cl_platform_id platfromID, cl_platform_info CL_PLATFORM_INFO, std::string* value)
{
	cl_int clStatus = 0;
	size_t infoBufferLength = 0;
	char* infoBuffer = NULL;

	clStatus = clGetPlatformInfo(platfromID, CL_PLATFORM_INFO, 0, NULL, &infoBufferLength);
	CHECK_CL(clStatus);

	infoBuffer = new char[infoBufferLength];
	clStatus = clGetPlatformInfo(platfromID, CL_PLATFORM_INFO, infoBufferLength, infoBuffer, NULL);
	CHECK_CL(clStatus);

	std::string ret = infoBuffer;
	delete[] infoBuffer;

	ret = trimString(ret);
	ret = reduceString(ret);

	*value = ret; 
}

// retrieve a single value of device info
template <typename T> void inline get_device_info(cl_device_id deviceID, cl_device_info CL_DEVICE_INFO, T* value)
{
	cl_int clStatus = 0;

	clStatus = clGetDeviceInfo(deviceID, CL_DEVICE_INFO, sizeof(T), value, NULL);
	CHECK_CL(clStatus);
}

// retrieve an array of values of device info
template <typename T> void inline get_device_info(cl_device_id deviceID, cl_device_info CL_DEVICE_INFO, T* values, ::size_t count)
{
	cl_int clStatus = 0;

	clStatus = clGetDeviceInfo(deviceID, CL_DEVICE_INFO, sizeof(T) * count, values, NULL);
	CHECK_CL(clStatus);
}

// retrieve a string value of device info
template <> void inline get_device_info(cl_device_id deviceID, cl_device_info CL_DEVICE_INFO, std::string* value)
{
	cl_int clStatus = 0;
	size_t infoBufferLength = 0;
	char* infoBuffer = NULL;

	clStatus = clGetDeviceInfo(deviceID, CL_DEVICE_INFO, 0, NULL, &infoBufferLength);
	CHECK_CL(clStatus);

	infoBuffer = new char[infoBufferLength];
	clStatus = clGetDeviceInfo(deviceID, CL_DEVICE_INFO, infoBufferLength, infoBuffer, NULL);
	CHECK_CL(clStatus);

	std::string ret = infoBuffer;
	delete[] infoBuffer;

	ret = trimString(ret);
	ret = reduceString(ret);

	*value = ret; 
}

inline std::vector<cl_context_properties> createContextProps(cl_platform_id selectedPlatformID, bool useInterop)
{
	std::vector<cl_context_properties> clContextProps;

#ifdef BUILD_WINDOWS
	{
		clContextProps.push_back( CL_CONTEXT_PLATFORM );
		clContextProps.push_back( (cl_context_properties)( selectedPlatformID ) );
		if(useInterop)
		{
			clContextProps.push_back( CL_GL_CONTEXT_KHR );
			clContextProps.push_back( (cl_context_properties)( wglGetCurrentContext() ) ) ;
			clContextProps.push_back( CL_WGL_HDC_KHR );
			clContextProps.push_back( (cl_context_properties)( wglGetCurrentDC() ) );
		}
		clContextProps.push_back(0);
	}
#endif

#ifdef BUILD_UNIX
	{
		clContextProps.push_back( CL_CONTEXT_PLATFORM );
		clContextProps.push_back( (cl_context_properties)( selectedPlatformID ) );
		if(useInterop)
		{
			clContextProps.push_back( CL_GL_CONTEXT_KHR );
			clContextProps.push_back( (cl_context_properties)( glXGetCurrentContext() ) );
			clContextProps.push_back( CL_GLX_DISPLAY_KHR );
			clContextProps.push_back( (cl_context_properties)( glXGetCurrentDisplay() ) );
		}
		clContextProps.push_back( 0 );
	};
#endif

#ifdef BUILD_APPLE
	// TODO
	CGLContextObj kCGLContext = CGLGetCurrentContext();
	CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);

	cl_context_properties clProperties[] =
	{
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup,
		0
	};
#endif

	return clContextProps;
}

class OpenCLDeviceInfo
{
public:
	OpenCLDeviceInfo() : platfromID(0), deviceID(0), isCurrentGLDevice(false), isCapableGLInterop(false) { }
	~OpenCLDeviceInfo() {}

	bool isExtSupported(const std::string& extString)
	{
		for(unsigned int ext = 0; ext < extensionTokens.size(); ++ext)
		{
			if(extString == extensionTokens[ext])
			{
				return true;
			};
		}

		return false;
	}

	bool isExtGLSharingSupported()
	{
		return isExtSupported("cl_khr_gl_sharing") || isExtSupported("cl_apple_gl_sharing") || isExtSupported("cl_APPLE_gl_sharing");
	}

	bool isCPU()
	{
		return (deviceType & CL_DEVICE_TYPE_CPU) > 0;
	}

	bool isGPU()
	{
		return (deviceType & CL_DEVICE_TYPE_GPU) > 0;
	}

	bool isACC()
	{
		return (deviceType & CL_DEVICE_TYPE_ACCELERATOR) > 0;
	}

	void checkInteropInfo()
	{
		bool hasInterop = isExtGLSharingSupported();
		
		if(hasInterop == false)
			return;

		std::vector<cl_context_properties> clContextPropsInterop = createContextProps(platfromID, true);
	
		cl_int clStatus = 0;

		if(clGetExtensionFunctionAddress)
		{
			*(void **)(&__GetGLContextInfoKHR) = clGetExtensionFunctionAddress("clGetGLContextInfoKHR");
		}
		else if(clGetExtensionFunctionAddressForPlatform)
		{
			*(void **)(&__GetGLContextInfoKHR) = clGetExtensionFunctionAddressForPlatform(platfromID, "clGetGLContextInfoKHR");
		}

		if(clGetGLContextInfoKHR)
		{
			cl_device_id interopDeviceID;
			clStatus = clGetGLContextInfoKHR(&clContextPropsInterop[0], CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, sizeof(cl_device_id), &interopDeviceID, NULL);

			if(clStatus == CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR)
				return;

			CHECK_CL(clStatus);

			isCurrentGLDevice = (interopDeviceID == deviceID);

			size_t interopDevicesSize = 0;
			clStatus = clGetGLContextInfoKHR(&clContextPropsInterop[0], CL_DEVICES_FOR_GL_CONTEXT_KHR, sizeof(cl_device_id), NULL, &interopDevicesSize);
			CHECK_CL(clStatus);

			size_t numInteropDevices = interopDevicesSize / sizeof(cl_device_id);

			if(numInteropDevices)
			{
				cl_device_id* interopDeviceIDs = new cl_device_id[numInteropDevices];
				clStatus = clGetGLContextInfoKHR(&clContextPropsInterop[0], CL_DEVICES_FOR_GL_CONTEXT_KHR, numInteropDevices*sizeof(cl_device_id), interopDeviceIDs, &interopDevicesSize);
				CHECK_CL(clStatus);

				for(size_t i = 0; i < numInteropDevices; ++i)
				{
					if(interopDeviceIDs[i] == deviceID)
					{
						isCapableGLInterop = true;
						break;
					}
				}

				delete[] interopDeviceIDs;
			}
		}
	}

	void init(cl_platform_id platfromID, cl_device_id deviceID)
	{
		this->platfromID = platfromID;
		this->deviceID = deviceID;

		get_device_info(deviceID, CL_DEVICE_VENDOR, &deviceVendor);
		get_device_info(deviceID, CL_DEVICE_NAME, &deviceName);
		get_device_info(deviceID, CL_DEVICE_VERSION, &deviceVersion);
		get_device_info(deviceID, CL_DRIVER_VERSION, &deviceDriverVersion);
		get_device_info(deviceID, CL_DEVICE_PROFILE, &deviceProfile);
		get_device_info(deviceID, CL_DEVICE_TYPE, &deviceType);

		std::string extensionStr;
		get_device_info(deviceID, CL_DEVICE_EXTENSIONS, &extensionStr);

		std::istringstream extensionSS(extensionStr);
		copy(std::istream_iterator<std::string>(extensionSS), std::istream_iterator<std::string>(), std::back_inserter <std::vector<std::string> >(extensionTokens));
		std::sort(extensionTokens.begin(), extensionTokens.end());

		get_device_info(deviceID, CL_DEVICE_MAX_CLOCK_FREQUENCY, &deviceMaxClockFreq);
		get_device_info(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, &deviceMaxComputeUnits);

		get_device_info(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, &deviceMaxWGSize);
		get_device_info(deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, deviceMaxWWISize, 3);

		if( isExtSupported("cl_nv_device_attribute_query") )
		{
			#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV       0x4000
			#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV       0x4001
			#define CL_DEVICE_REGISTERS_PER_BLOCK_NV            0x4002
			#define CL_DEVICE_WARP_SIZE_NV                      0x4003
			#define CL_DEVICE_GPU_OVERLAP_NV                    0x4004
			#define CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV            0x4005
			#define CL_DEVICE_INTEGRATED_MEMORY_NV              0x4006

			get_device_info(deviceID, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, &nvCCMajor);
			get_device_info(deviceID, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, &nvCCMinor);
			get_device_info(deviceID, CL_DEVICE_REGISTERS_PER_BLOCK_NV, &nvRegBlock);
			get_device_info(deviceID, CL_DEVICE_WARP_SIZE_NV, &nvWarpSize);
			get_device_info(deviceID, CL_DEVICE_GPU_OVERLAP_NV, &nvOverlap);
			get_device_info(deviceID, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, &nvExecTimeout);
			get_device_info(deviceID, CL_DEVICE_INTEGRATED_MEMORY_NV, &nvUnifiedMem);
		}

		checkInteropInfo();
	}

	void print()
	{
		std::cout << tabulateStrings("CL_DEVICE_VENDOR:", deviceVendor) << std::endl;
		std::cout << tabulateStrings("CL_DEVICE_NAME:", deviceName) << std::endl;
		std::cout << tabulateStrings("CL_DEVICE_VERSION:", deviceVersion) << std::endl;
		std::cout << tabulateStrings("CL_DRIVER_VERSION:", deviceDriverVersion) << std::endl;
		std::cout << tabulateStrings("CL_DEVICE_PROFILE:", deviceProfile) << std::endl;

		std::cout << tabulateStrings("CL_DEVICE_MAX_CLOCK_FREQUENCY:", deviceMaxClockFreq) << std::endl;
		std::cout << tabulateStrings("CL_DEVICE_MAX_COMPUTE_UNITS:", deviceMaxComputeUnits) << std::endl;
		std::cout << tabulateStrings("CL_DEVICE_MAX_WORK_GROUP_SIZE:", deviceMaxWGSize) << std::endl;

		std::stringstream ss;
		ss << deviceMaxWWISize[0] << "x" << deviceMaxWWISize[1] << "x" << deviceMaxWWISize[2];
		std::cout << tabulateStrings("CL_DEVICE_MAX_WORK_ITEM_SIZES:", ss.str()) << std::endl;

		if( isExtSupported("cl_nv_device_attribute_query") )
		{
			std::stringstream ss;
			ss << nvCCMajor << "." << nvCCMinor;
			std::cout << tabulateStrings("CL_DEVICE_CC_NV: ", ss.str()) << std::endl;

			//std::cout << tabulateStrings("CL_DEVICE_CC_MAJOR_NV:", nvCCMajor) << std::endl;
			//std::cout << tabulateStrings("CL_DEVICE_CC_MINOR_NV:", nvCCMinor) << std::endl;
			//std::cout << tabulateStrings("CL_DEVICE_REG_BLOCK_NV:", nvRegBlock) << std::endl;
			//std::cout << tabulateStrings("CL_DEVICE_WARP_SIZE_NV:", nvWarpSize) << std::endl;
			//std::cout << tabulateStrings("CL_DEVICE_GPU_OVERLAP_NV:", nvOverlap) << std::endl;
			//std::cout << tabulateStrings("CL_DEVICE_EXEC_TIMEOUT_NV:", nvExecTimeout) << std::endl;
			//std::cout << tabulateStrings("CL_DEVICE_INTEG_MEM_NV:", nvUnifiedMem) << std::endl;
		}

		std::cout << "CL_DEVICE_EXTENSIONS:" << std::endl;
		for(unsigned int ext = 0; ext < extensionTokens.size(); ++ext)
		{
			std::cout << tabulateStrings("", extensionTokens[ext]) << std::endl;
		}
	}

	std::string deviceVendor;
	std::string deviceName;
	std::string deviceVersion;
	std::string deviceDriverVersion;
	std::string deviceProfile;

	cl_uint deviceMaxClockFreq;
	cl_uint deviceMaxComputeUnits;

	size_t deviceMaxWGSize;
	size_t deviceMaxWWISize[3];

	std::vector<std::string> extensionTokens;

	cl_device_type deviceType;

	// nv only
	cl_uint nvCCMajor;
	cl_uint nvCCMinor;
	cl_uint nvRegBlock;
	cl_uint nvWarpSize;
	cl_bool nvOverlap;
	cl_bool nvExecTimeout;
	cl_bool nvUnifiedMem;

	cl_platform_id platfromID;
	cl_device_id deviceID;

	bool isCurrentGLDevice;
	bool isCapableGLInterop;
};

class OpenCLPlatformInfo
{
public:
	OpenCLPlatformInfo() : platfromID(0), numDevices(0), deviceIDs(NULL), devices(NULL) { }
	~OpenCLPlatformInfo() 
	{
		DEALLOC_ARR(deviceIDs);
		DEALLOC_ARR(devices);
	}

	bool isExtSupported(const std::string& extString)
	{
		for(unsigned int ext = 0; ext < extensionTokens.size(); ++ext)
		{
			if(extString == extensionTokens[ext])
			{
				return true;
			};
		}

		return false;
	}

	void init(cl_platform_id platfromID)
	{
		this->platfromID = platfromID;

		get_platform_info(platfromID, CL_PLATFORM_VENDOR, &platformVendor);
		get_platform_info(platfromID, CL_PLATFORM_NAME, &platformName);
		get_platform_info(platfromID, CL_PLATFORM_VERSION, &platformVersion);
		get_platform_info(platfromID, CL_PLATFORM_PROFILE, &platformProfile);

		std::string extensionStr;
		get_platform_info(platfromID, CL_PLATFORM_EXTENSIONS, &extensionStr);

		std::istringstream extensionSS(extensionStr);
		copy(std::istream_iterator<std::string>(extensionSS), std::istream_iterator<std::string>(), std::back_inserter <std::vector<std::string> >(extensionTokens));
		std::sort(extensionTokens.begin(), extensionTokens.end());

		cl_int clStatus = 0;

		clStatus = clGetDeviceIDs( platfromID, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices );
		CHECK_CL(clStatus);

		deviceIDs = new cl_device_id[numDevices];
		clStatus = clGetDeviceIDs( platfromID, CL_DEVICE_TYPE_ALL, numDevices, deviceIDs, &numDevices );
		CHECK_CL(clStatus);

		devices = new OpenCLDeviceInfo[numDevices];
		for(cl_uint d = 0; d < numDevices; ++d)
		{
			devices[d].init(platfromID, deviceIDs[d]);
		}
	}

	void printPlatform()
	{
		std::cout << tabulateStrings("CL_PLATFORM_VENDOR:", platformVendor) << std::endl;
		std::cout << tabulateStrings("CL_PLATFORM_NAME:", platformName) << std::endl;
		std::cout << tabulateStrings("CL_PLATFORM_VERSION:", platformVersion) << std::endl;
		std::cout << tabulateStrings("CL_PLATFORM_PROFILE:", platformProfile) << std::endl;
	}

	void printDevice()
	{
		for(cl_uint d = 0; d < numDevices; ++d)
		{
			std::cout << "DEVICE #" << d << std::endl;
			devices[d].print();
			std::cout << std::endl;
		}
	}

	void print()
	{
		printPlatform();
		std::cout << std::endl;
		printDevice();
		std::cout << std::endl;
	}

	bool isCL11()
	{
		return platformVersion.find("OpenCL 1.1") != std::string::npos;
	}

	bool isCL12()
	{
		return platformVersion.find("OpenCL 1.2") != std::string::npos;
	}

	bool isCL20()
	{
		return platformVersion.find("OpenCL 2.0") != std::string::npos;
	}

	cl_platform_id platfromID;

	cl_uint numDevices;
	cl_device_id* deviceIDs;
	OpenCLDeviceInfo* devices;

	std::string platformVendor;
	std::string platformName;
	std::string platformVersion;
	std::string platformProfile;

	std::vector<std::string> extensionTokens;
};

class OpenCLUtil
{
public:
	OpenCLUtil() : numPlatforms(0), platfromIDs(NULL), platforms(NULL) { }
	~OpenCLUtil()
	{
		DEALLOC_ARR(platfromIDs);
		DEALLOC_ARR(platforms);
	}

	void init()
	{
		cl_int clStatus = 0;

		clStatus = clGetPlatformIDs( 0, NULL, &numPlatforms ); 
		CHECK_CL(clStatus);

		platfromIDs = new cl_platform_id[numPlatforms];
		clStatus = clGetPlatformIDs( numPlatforms, platfromIDs, NULL );
		CHECK_CL(clStatus);

		platforms = new OpenCLPlatformInfo[numPlatforms];
		for(cl_uint p = 0; p < numPlatforms; ++p)
		{
			platforms[p].init(platfromIDs[p]);
		}
	}

	void print()
	{
		for(cl_uint p = 0; p < numPlatforms; ++p)
		{
			std::cout << "PLATFORM #" << p << std::endl;
			platforms[p].print();
			std::cout << std::endl;
		}
	}

	void selectePlatformDevice(cl_uint& selectedPlatformIndex, cl_uint& selectedDeviceIndex, cl_platform_id& selectedPlatformID, cl_device_id& selectedDeviceID)
	{
		if(selectedPlatformIndex != -1 && selectedDeviceIndex != -1)
		{
			if(selectedPlatformIndex >= numPlatforms || selectedDeviceIndex >= platforms[selectedPlatformIndex].numDevices)
			{
				// err throw exception
			}
			else
			{
				std::cout << "PLATFORM #" << selectedPlatformIndex << std::endl;
				platforms[selectedPlatformIndex].printPlatform();
				std::cout << std::endl;

				std::cout << "DEVICE #" << selectedDeviceIndex << std::endl;
				platforms[selectedPlatformIndex].devices[selectedDeviceIndex].print();
				std::cout << std::endl;

				selectedPlatformID = platfromIDs[selectedPlatformIndex];
				selectedDeviceID = platforms[selectedPlatformIndex].deviceIDs[selectedDeviceIndex];
			}

			return;
		}

		for(cl_uint p = 0; p < numPlatforms; ++p)
		{
			std::cout << "PLATFORM #" << p << std::endl;
			platforms[p].printPlatform();
			std::cout << std::endl;
		}

		bool platformReadOK = false;
		while(selectedPlatformIndex >= numPlatforms && !platformReadOK)
		{
			std::cout << "Select platform: ";
			platformReadOK = !(std::cin >> selectedPlatformIndex);
			std::cin.clear();
			std::cout << std::endl;
		}
		selectedPlatformID = platfromIDs[selectedPlatformIndex];

		platforms[selectedPlatformIndex].printDevice();

		bool deviceReadOK = false;
		while(selectedDeviceIndex >= platforms[selectedPlatformIndex].numDevices && !deviceReadOK)
		{
			std::cout << "Select device: ";
			deviceReadOK = !(std::cin >> selectedDeviceIndex);
			std::cin.clear();
			std::cout << std::endl;
		}
		selectedDeviceID = platforms[selectedPlatformIndex].deviceIDs[selectedDeviceIndex];
	}

	void selectePlatformInteropDevice(cl_uint& selectedPlatformIndex, cl_uint& selectedDeviceIndex, cl_platform_id& selectedPlatformID, cl_device_id& selectedDeviceID)
	{
		for(cl_uint p = 0; p < numPlatforms; ++p)
		{
			for(cl_uint d = 0; d < platforms[p].numDevices; ++d)
			{
				if(platforms[p].devices[d].isCurrentGLDevice == true)
				{
					selectedPlatformIndex = p;
					selectedDeviceIndex = d;

					selectedPlatformID = platforms[p].platfromID;
					selectedDeviceID = platforms[p].devices[d].deviceID;

					std::cout << "PLATFORM #" << p << std::endl;
					platforms[p].printPlatform();
					std::cout << std::endl;

					std::cout << "DEVICE #" << d << std::endl;
					platforms[p].devices[d].print();
					std::cout << std::endl;
				}
			}
		}
	}

	cl_uint numPlatforms;
	cl_platform_id* platfromIDs;
	OpenCLPlatformInfo* platforms;
};

#endif