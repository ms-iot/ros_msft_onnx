cmake_minimum_required(VERSION 3.5)

project(object_tracker)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(onnxruntime_vendor REQUIRED)

# The node expects to use the Tiny YOLO model available in the ONNX model zoo.
file(DOWNLOAD
  https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx
  ${CMAKE_CURRENT_BINARY_DIR}/testdata/model.onnx
  SHOW_PROGRESS
)

add_executable(object_tracker src/main.cpp)
ament_target_dependencies(object_tracker rclcpp std_msgs onnxruntime_vendor)

install(TARGETS
  object_tracker
  DESTINATION lib/${PROJECT_NAME})

ament_package()