project(neural_network)

# Specify default build type if none provided (before project() command)
set(CMAKE_BUILD_TYPE Debug)
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

# For generators with multiple configurations (like VS), only allow Debug and Release
if(CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Limited configurations" FORCE)
	mark_as_advanced(CMAKE_CONFIGURATION_TYPES)
endif()

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_NAME "neural_network")

if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Werror")
endif()
if (MSVC)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W4 -Werror")
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE sources      src/*.cpp)
add_executable(neural_network ${sources})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
