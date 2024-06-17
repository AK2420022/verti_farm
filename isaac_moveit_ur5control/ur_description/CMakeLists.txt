cmake_minimum_required(VERSION 3.10.2)
project(ur_description)
find_package(ament_cmake REQUIRED)

ament_package()

install(DIRECTORY config meshes urdf DESTINATION share/${PROJECT_NAME})
