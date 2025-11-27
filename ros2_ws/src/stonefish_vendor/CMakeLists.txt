cmake_minimum_required(VERSION 3.8)
project(stonefish_vendor)

find_package(ament_cmake REQUIRED)

set(STONEFISH_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/stonefish)

# Build Stonefish 
add_subdirectory(${STONEFISH_ROOT_DIR}
                 ${CMAKE_CURRENT_BINARY_DIR}/stonefish_build)

# Install headers
install(DIRECTORY ${STONEFISH_ROOT_DIR}/Library/include/
        DESTINATION include/Stonefish)

# Install libraries
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/stonefish_build/
        DESTINATION lib
        FILES_MATCHING PATTERN "*.so*")

# We write this file to explicitly link the installed library to the installed headers
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
    INTERFACE_INCLUDE_DIRECTORIES \"${CMAKE_INSTALL_PREFIX}/include\"
  )
endif()
")

# Overwrite upstream config for CI
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/StonefishConfig.cmake
        DESTINATION lib/cmake/Stonefish)

ament_package()