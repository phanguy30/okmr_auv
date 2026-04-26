cmake_minimum_required(VERSION 3.8)
project(okmr_controls)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(okmr_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(include)

# Create PID controller library
add_library(pid_controller
  src/pid_controller.cpp
)
ament_target_dependencies(pid_controller rclcpp)

# Create control layer base library
add_library(control_layer_base
  src/control_layer_base.cpp
)
ament_target_dependencies(control_layer_base 
  rclcpp 
  geometry_msgs 
  okmr_msgs
)
target_link_libraries(control_layer_base pid_controller)

# Create velocity control layer executable
add_executable(velocity_control_layer_node
  src/velocity_control_layer.cpp
)
ament_target_dependencies(velocity_control_layer_node
  rclcpp
  geometry_msgs
  okmr_msgs
)
target_link_libraries(velocity_control_layer_node control_layer_base)

# Create pose control layer executable
add_executable(pose_control_layer_node
  src/pose_control_layer.cpp
)
ament_target_dependencies(pose_control_layer_node
  rclcpp
  geometry_msgs
  okmr_msgs
)
target_link_libraries(pose_control_layer_node control_layer_base)

# Create acceleration control layer executable
add_executable(accel_control_layer_node
  src/accel_control_layer.cpp
)
ament_target_dependencies(accel_control_layer_node
  rclcpp
  geometry_msgs
  okmr_msgs
)
target_link_libraries(accel_control_layer_node control_layer_base)

# Create thrust allocator executable
add_executable(thrust_allocator_node
  src/thrust_allocator.cpp
)
ament_target_dependencies(thrust_allocator_node
  rclcpp
  geometry_msgs
  okmr_msgs
  Eigen3
)
target_link_libraries(thrust_allocator_node Eigen3::Eigen)

ament_python_install_package(${PROJECT_NAME})

# Install C++ executables
install(TARGETS
  velocity_control_layer_node
  pose_control_layer_node
  accel_control_layer_node
  thrust_allocator_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install libraries
install(TARGETS
  pid_controller
  control_layer_base
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install header files
install(DIRECTORY include/
  DESTINATION include
)

# Export libraries
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp geometry_msgs okmr_msgs)

#install python execs
install(PROGRAMS
  okmr_controls/motor_thrust_test.py
  okmr_controls/pose_test.py
  okmr_controls/velocity_test.py
  okmr_controls/accel_test.py
  okmr_controls/enable_and_control.py
  okmr_controls/goal_velocity_test.py
  okmr_controls/wrench_test.py
  DESTINATION lib/${PROJECT_NAME}
)
#install launch and params

install(DIRECTORY
  launch
  params
  gui_config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  # Integration tests
  find_package(ament_cmake_ros REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  function(add_ros_isolated_launch_test path)
    set(RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
    add_launch_test("${path}" RUNNER "${RUNNER}" ${ARGN})
  endfunction()
  #add_ros_isolated_launch_test(test/file_name.py)
endif()

ament_package()
