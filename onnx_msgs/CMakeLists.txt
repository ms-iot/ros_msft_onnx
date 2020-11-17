# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.5)
project(onnx_msgs)

find_package(ament_cmake REQUIRED) # TODO: Needed?
find_package(rclcpp REQUIRED)

find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedObjectPose.msg"
  DEPENDENCIES std_msgs geometry_msgs
 )

ament_package()