cmake_minimum_required(VERSION 3.8)
project(robot_moveit)

# set C++ standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- dependencies ---
set(DEPENDENCIES
    ament_cmake
    rclcpp
    rclcpp_action
    geometry_msgs
    ur_msgs
    control_msgs
    controller_manager_msgs
    std_srvs
    moveit_ros_planning_interface
    moveit_visual_tools
    tf2
    tf2_ros
    tf2_geometry_msgs
    grasp_interfaces
  )
  
foreach(dep ${DEPENDENCIES})
  find_package(${dep} REQUIRED)
endforeach()

# --- executables ---

add_executable(waypoints src/waypoints.cpp)
target_include_directories(waypoints PRIVATE include)
ament_target_dependencies(waypoints ${DEPENDENCIES})
install(TARGETS waypoints DESTINATION lib/${PROJECT_NAME})

add_executable(ik src/ik.cpp)
target_include_directories(ik PRIVATE include)
ament_target_dependencies(ik ${DEPENDENCIES})
install(TARGETS ik DESTINATION lib/${PROJECT_NAME})

add_executable(task src/task.cpp src/task_executor.cpp src/task_executor_utils.cpp)
target_include_directories(task PRIVATE include)
ament_target_dependencies(task ${DEPENDENCIES})
target_link_libraries(task tf2_geometry_msgs::tf2_geometry_msgs)
install(TARGETS task DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
