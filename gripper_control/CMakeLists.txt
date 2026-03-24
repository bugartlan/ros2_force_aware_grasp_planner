cmake_minimum_required(VERSION 3.8)
project(gripper_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ur_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_executable(gripper_control src/wrapper.cpp)
ament_target_dependencies(gripper_control rclcpp ur_msgs std_srvs)
install(TARGETS gripper_control DESTINATION lib/${PROJECT_NAME})

ament_package()
