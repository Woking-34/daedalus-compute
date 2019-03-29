cmake_minimum_required(VERSION 2.8.11)
cmake_policy(VERSION 2.8.11)

project(Daedalus-Compute)

set( CMAKE_VERBOSE_MAKEFILE ON )
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "_cmake")

IF(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
ENDIF()

string(TOUPPER ${CMAKE_CXX_COMPILER_ID} CXX_COMPILER_ID_UPPER)
if(${CXX_COMPILER_ID_UPPER} STREQUAL MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /MP")
endif()

IF(WIN32)
	ADD_DEFINITIONS( -DBUILD_WINDOWS )
	
	ADD_DEFINITIONS(-DNOMINMAX)
	ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
	ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)
	
	IF(CMAKE_CL_64)
		ADD_DEFINITIONS( -DBUILD_WINDOWS64 )
		
		SET(EXECUTABLE_OUTPUT_PATH
		${PROJECT_SOURCE_DIR}/daedalus-bin/Win64/
		)
		SET(LIBRARY_OUTPUT_PATH
			${PROJECT_SOURCE_DIR}/daedalus-lib/Win64/
		)
		
		link_directories( daedalus-lib/Win64/${CMAKE_BUILD_TYPE} )
	ELSE()
		ADD_DEFINITIONS( -DBUILD_WINDOWS32 )
		
		SET(EXECUTABLE_OUTPUT_PATH
		${PROJECT_SOURCE_DIR}/daedalus-bin/Win32/
		)
		SET(LIBRARY_OUTPUT_PATH
			${PROJECT_SOURCE_DIR}/daedalus-lib/Win32/
		)
		
		link_directories( daedalus-lib/Win32/${CMAKE_BUILD_TYPE} )
	ENDIF()
ENDIF(WIN32)

IF(UNIX AND NOT APPLE)
	ADD_DEFINITIONS( -DBUILD_UNIX )
	
	SET(EXECUTABLE_OUTPUT_PATH
		${PROJECT_SOURCE_DIR}/daedalus-bin/linux/${CMAKE_BUILD_TYPE}
	)
	SET(LIBRARY_OUTPUT_PATH
		${PROJECT_SOURCE_DIR}/daedalus-lib/linux/${CMAKE_BUILD_TYPE}
	)
ENDIF(UNIX AND NOT APPLE)

include_directories(
	./inc
	./src_fw
)

link_directories(
	./lib/${CMAKE_BUILD_TYPE}
)

#IF(WIN32)
#	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
#	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
#ENDIF(WIN32)
#
#IF(UNIX AND NOT APPLE)
#	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#ENDIF(UNIX AND NOT APPLE)

#--------------------------------------------------------------------
find_package(CUDA)
IF(CUDA_FOUND)
	MESSAGE(STATUS "CUDA FOUND!")
	MESSAGE(STATUS "CUDA DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
ELSE()
	MESSAGE(STATUS "CUDA NOT FOUND!")
ENDIF()
#--------------------------------------------------------------------

IF(WIN32)
	ADD_SUBDIRECTORY( ./src_lib/freeglut )
ENDIF(WIN32)
ADD_SUBDIRECTORY( ./src_lib/glew )
ADD_SUBDIRECTORY( ./src_lib/clew )

include( ./CMakeUtils.txt )

include( ./src_fw/appfw/CMakeTokens.txt )

include( ./src_fw/system/CMakeTokens.txt )
include( ./src_fw/math/CMakeTokens.txt )
include( ./src_fw/assets/CMakeTokens.txt )

include( ./src_fw/glutil/CMakeTokens.txt )
include( ./src_fw/clutil/CMakeTokens.txt )
include( ./src_fw/cuutil/CMakeTokens.txt )

ADD_SUBDIRECTORY( ./src_fw/applications/comp_deform )
ADD_SUBDIRECTORY( ./src_fw/applications/comp_cloth )
ADD_SUBDIRECTORY( ./src_fw/applications/comp_particles )

ADD_SUBDIRECTORY( ./src_fw/applications/compute_info )

#ADD_SUBDIRECTORY( ./src_fw/applications/gl_mesh_seq )
#ADD_SUBDIRECTORY( ./src_fw/applications/gl_mesh_data )
#ADD_SUBDIRECTORY( ./src_fw/applications/gl_volume_viz 
