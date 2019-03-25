ADD_DEFINITIONS(-DGLEW_STATIC)
ADD_DEFINITIONS(-DFREEGLUT_STATIC)

ADD_EXECUTABLE( compute_info_opencl compute_info_opencl.cpp ${SRC_SYSTEM} ${SRC_CLUTIL} ${SRC_GLUTIL} )

IF(WIN32)
	TARGET_LINK_LIBRARIES( compute_info_opencl clew glew freeglut )
ENDIF(WIN32)

IF(UNIX AND NOT APPLE)
	TARGET_LINK_LIBRARIES( compute_info_opencl clew glew glut GL )
ENDIF(UNIX AND NOT APPLE)

source_group("compute_info_opencl\\system" FILES ${SRC_SYSTEM})
source_group("compute_info_opencl\\glutil\\" FILES ${SRC_GLUTIL})
source_group("compute_info_opencl\\clutil\\" FILES ${SRC_CLUTIL})