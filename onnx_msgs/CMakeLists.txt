# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.5)
project(onnx_msgs)

find_package(ament_cmake REQUIRED) # TODO: Needed?
find_package(rclcpp REQUIRED)

find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)


rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedObjectPose.msg"
 )

ament_package()