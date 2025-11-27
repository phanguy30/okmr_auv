cmake_minimum_required(VERSION 3.8)
project(stonefish_vendor)

find_package(ament_cmake REQUIRED)

find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(Freetype REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Bullet REQUIRED)

set(STONEFISH_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/stonefish)

add_subdirectory(${STONEFISH_ROOT_DIR}
                 ${CMAKE_CURRENT_BINARY_DIR}/stonefish_build)

target_include_directories(Stonefish PUBLIC
  $<BUILD_INTERFACE:${STONEFISH_ROOT_DIR}/Library/include>
  $<INSTALL_INTERFACE:include> 
  $<INSTALL_INTERFACE:include/Stonefish> 
)

install(DIRECTORY ${STONEFISH_ROOT_DIR}/Library/include/
        DESTINATION include/Stonefish)

install(TARGETS Stonefish
        EXPORT StonefishTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

install(EXPORT StonefishTargets
        FILE StonefishTargets.cmake
        NAMESPACE Stonefish::
        DESTINATION share/Stonefish/cmake)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/StonefishConfig.cmake
"
include(CMakeFindDependencyMacro)

find_dependency(OpenGL)
find_dependency(SDL2)
find_dependency(Freetype)
find_dependency(OpenMP)
find_dependency(Bullet)

include(\"\${CMAKE_CURRENT_LIST_DIR}/StonefishTargets.cmake\")
")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/StonefishConfig.cmake
        DESTINATION share/Stonefish/cmake)

ament_export_dependencies(OpenGL SDL2 Freetype OpenMP Bullet)

ament_package()