SET(SRC_COMP_RTOW
	${CMAKE_CURRENT_SOURCE_DIR}/App_comp_rtow.h
	${CMAKE_CURRENT_SOURCE_DIR}/App_comp_rtow.cpp
	
	${CMAKE_CURRENT_SOURCE_DIR}/COMP_RTOW.h
	${CMAKE_CURRENT_SOURCE_DIR}/COMP_RTOW.cpp
	
	${CMAKE_CURRENT_SOURCE_DIR}/COMP_RTOW_TYPES.h
)

SET(SRC_COMPCL_RTOW
	${SRC_COMP_RTOW}
	
	${CMAKE_CURRENT_SOURCE_DIR}/COMPCL_RTOW.h
	${CMAKE_CURRENT_SOURCE_DIR}/COMPCL_RTOW.cpp
)

IF(CUDA_FOUND)
	FILE(GLOB_RECURSE CUFILES "*.cu")
	
	SET(SRC_COMPCU_RTOW
		${SRC_COMP_RTOW}
		
		${CMAKE_CURRENT_SOURCE_DIR}/COMPCU_RTOW.h
		${CMAKE_CURRENT_SOURCE_DIR}/COMPCU_RTOW.cpp
		
		${CUFILES}
	)
	
	#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
	#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
	#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_32,code=sm_32)
	#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
	#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
	
	LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
	LIST(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")

	IF(NOT ANDROID)
		#SET(CUDA_GENERATED_OUTPUT_DIR ${SOURCE_DIR}/bin)

		SET(COMPILE_TO_PTX  ON)
		CUDA_COMPILE_PTX(PTXFILES ${CUFILES})
	ENDIF()
ENDIF(CUDA_FOUND)

#ADD_SUBDIRECTORY( ./comp_gl )
ADD_SUBDIRECTORY( ./comp_cl )
ADD_SUBDIRECTORY( ./comp_cu )