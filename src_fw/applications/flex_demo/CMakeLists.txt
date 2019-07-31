macro(create_flex APP_NAME)
	ADD_DEFINITIONS(-DGLEW_STATIC)
	ADD_DEFINITIONS(-DFREEGLUT_STATIC)
	
	ADD_EXECUTABLE( ${APP_NAME} ${SRC_APP_FLEX} ${APP_SRC} ${SRC_APPFW} ${SRC_SYSTEM} ${SRC_MATH} ${SRC_ASSETS} ${SRC_GLUTIL} )
	
	IF(WIN32)
		TARGET_LINK_LIBRARIES( ${APP_NAME} ${APP_LIBS} glew freeglut )
	ENDIF(WIN32)
		
	IF(UNIX AND NOT APPLE)
		TARGET_LINK_LIBRARIES( ${APP_NAME} ${APP_LIBS} glew glut GL )
	ENDIF(UNIX AND NOT APPLE)
	
	target_include_directories(${APP_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
	target_link_libraries(${APP_NAME} freeglut)
	target_link_libraries(${APP_NAME} glew)
	
	target_include_directories(${APP_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../Flex )
	
	target_link_libraries(${APP_NAME} debug ${CMAKE_CURRENT_SOURCE_DIR}/../Flex/lib/win64/NvFlexDebugCUDA_x64.lib)
	target_link_libraries(${APP_NAME} debug ${CMAKE_CURRENT_SOURCE_DIR}/../Flex/lib/win64/NvFlexDeviceDebug_x64.lib)
	target_link_libraries(${APP_NAME} debug ${CMAKE_CURRENT_SOURCE_DIR}/../Flex/lib/win64/NvFlexExtDebugCUDA_x64.lib)
	
	target_link_libraries(${APP_NAME} optimized ${CMAKE_CURRENT_SOURCE_DIR}/../Flex/lib/win64/NvFlexReleaseCUDA_x64.lib)
	target_link_libraries(${APP_NAME} optimized ${CMAKE_CURRENT_SOURCE_DIR}/../Flex/lib/win64/NvFlexDeviceRelease_x64.lib)
	target_link_libraries(${APP_NAME} optimized ${CMAKE_CURRENT_SOURCE_DIR}/../Flex/lib/win64/NvFlexExtReleaseCUDA_x64.lib)
	
	MESSAGE(STATUS "Project added: " ${APP_NAME})
	
	source_group("${APP_NAME}" FILES ${APP_SRC})
		
	source_group("${APP_NAME}\\appfw" FILES ${SRC_APPFW})
	
	source_group("${APP_NAME}\\system" FILES ${SRC_SYSTEM})
	
	source_group("${APP_NAME}\\math\\" FILES ${SRC_MATH_BASE})
	source_group("${APP_NAME}\\math\\core" FILES ${SRC_MATH_CORE})
	source_group("${APP_NAME}\\math\\util" FILES ${SRC_MATH_UTIL})
	
	source_group("${APP_NAME}\\assets\\" FILES ${SRC_ASSETS})
	
	source_group("${APP_NAME}\\glutil\\" FILES ${SRC_GLUTIL})

	target_compile_definitions(${APP_NAME} PRIVATE ${APP_NAME})
endmacro()

SET(SRC_APP_FLEX
	../App_flex.h
	../App_flex.cpp

	../maths.h
	../maths.cpp

	../core.h
	../core.cpp

	../vec2.h
	../vec3.h
	../vec4.h

	../perlin.h
	../perlin.cpp
)

ADD_SUBDIRECTORY(flex_dambreak)
ADD_SUBDIRECTORY(flex_flagcloth)