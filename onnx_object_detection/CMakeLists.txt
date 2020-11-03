# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.5)

project(onnx_object_detection)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(MSVC)
  add_compile_options(/wd4251 /wd4275)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(onnxruntime_vendor REQUIRED)

# The node expects to use the Tiny YOLO model available in the ONNX model zoo.
file(DOWNLOAD
  https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx
  ${CMAKE_CURRENT_BINARY_DIR}/models/tinyyolov2-8.onnx
  SHOW_PROGRESS
)

include_directories(
  include
)

add_executable(object_detection
  src/main.cpp
  src/yolo_processor.cpp)

ament_target_dependencies(object_detection
  ament_index_cpp
  rclcpp
  std_msgs
  sensor_msgs
  visualization_msgs
  cv_bridge
  onnxruntime_vendor)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/models/tinyyolov2-8.onnx
  DESTINATION share/${PROJECT_NAME}/models)

install(TARGETS
  object_detection
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY
  rviz
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
