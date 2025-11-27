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

ament_package()