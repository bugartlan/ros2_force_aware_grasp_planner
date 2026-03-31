cmake_minimum_required(VERSION 3.8)
project(planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(PROGRAMS src/planner.py src/dummy.py src/GraspOptimizer.py src/grasp.py
                 src/optimizer.py src/config.py src/sampler.py
        DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY src/meshgraphnet DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY assets/ DESTINATION share/${PROJECT_NAME}/assets)

ament_package()
