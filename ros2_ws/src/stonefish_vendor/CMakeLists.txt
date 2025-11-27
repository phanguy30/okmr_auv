cmake_minimum_required(VERSION 3.8)
project(stonefish_vendor)

find_package(ament_cmake REQUIRED)

find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(Freetype REQUIRED)
find_package(OpenMP REQUIRED)

set(STONEFISH_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/stonefish)

add_subdirectory(${STONEFISH_ROOT_DIR}
                 ${CMAKE_CURRENT_BINARY_DIR}/stonefish_build)

install(DIRECTORY ${STONEFISH_ROOT_DIR}/Library/include/
        DESTINATION include/Stonefish)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/stonefish_build/version.h
        DESTINATION include/Stonefish)

install(DIRECTORY ${STONEFISH_ROOT_DIR}/3rdparty/
        DESTINATION include/Stonefish
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
        PATTERN "*.c" EXCLUDE PATTERN "*.cpp" EXCLUDE)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/stonefish_build/
        DESTINATION lib
        FILES_MATCHING PATTERN "libStonefish.so*")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/StonefishConfig.cmake
"
include(CMakeFindDependencyMacro)

find_dependency(OpenGL)
find_dependency(SDL2)
find_dependency(Freetype)
find_dependency(OpenMP)

if(NOT TARGET Stonefish::Stonefish)
  add_library(Stonefish::Stonefish SHARED IMPORTED)
  
  set_target_properties(Stonefish::Stonefish PROPERTIES
    IMPORTED_LOCATION \"${CMAKE_INSTALL_PREFIX}/lib/libStonefish.so\"
    INTERFACE_INCLUDE_DIRECTORIES \"${CMAKE_INSTALL_PREFIX}/include;${CMAKE_INSTALL_PREFIX}/include/Stonefish\"
  )
endif()
")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/StonefishConfig.cmake
        DESTINATION lib/cmake/Stonefish)

ament_export_dependencies(OpenGL SDL2 Freetype OpenMP)

ament_package()