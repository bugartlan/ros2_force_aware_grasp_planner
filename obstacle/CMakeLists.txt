cmake_minimum_required(VERSION 3.8)
project(obstacle)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_visual_tools REQUIRED)
find_package(control_msgs REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(ur_msgs REQUIRED)

add_executable(obstacle src/obstacle.cpp)
target_include_directories(obstacle PRIVATE include)
ament_target_dependencies(obstacle
  rclcpp
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  moveit_ros_planning_interface
  moveit_visual_tools
  ur_msgs
)
install(TARGETS obstacle DESTINATION lib/${PROJECT_NAME})

add_executable(force src/force.cpp)
target_include_directories(force PRIVATE include)
ament_target_dependencies(force
  rclcpp
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  control_msgs
  controller_manager_msgs
  ur_msgs
)
install(TARGETS force DESTINATION lib/${PROJECT_NAME})

ament_package()
