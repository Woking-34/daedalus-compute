cmake_minimum_required(VERSION 2.8.11)
cmake_policy(VERSION 2.8.11)

project(clew)

SET(SRC_CLEW
	clew.cpp
	clew.h
)

if(CMAKE_COMPILER_IS_GNUCC)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fno-common -fPIC -Wall")
endif()

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR})
ADD_LIBRARY( clew STATIC ${SRC_CLEW} )
SET_PROPERTY(TARGET clew PROPERTY FOLDER "_library")

IF(UNIX AND NOT APPLE)
	TARGET_LINK_LIBRARIES( clew dl )
ENDIF(UNIX AND NOT APPLE)