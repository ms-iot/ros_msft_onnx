cmake_minimum_required(VERSION 2.8.3)
project(ros_msft_onnx)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

#TODO: add compile defs

find_package(Eigen3 REQUIRED)

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  visualization_msgs
  ros_msft_onnx_msgs
  image_transport
  roscpp
  cv_bridge
  tf
  onnxruntime_vendor
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS std_msgs geometry_msgs visualization_msgs ros_msft_onnx_msgs image_transport roscpp cv_bridge tf onnxruntime_vendor
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${onnxruntime_vendor_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/ros_msft_onnx.cpp src/main.cpp src/yolo_box.cpp src/pose_parser.cpp)
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} windowsapp)

# The node expects to use the Tiny YOLO model available in the ONNX model zoo.
file(DOWNLOAD
  https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx
  ${CMAKE_CURRENT_BINARY_DIR}/testdata/model.onnx
  SHOW_PROGRESS
)

install(FILES
  $<TARGET_FILE:onnxruntime-unofficial::onnxruntime>
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch testdata ${CMAKE_CURRENT_BINARY_DIR}/testdata
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(catkin REQUIRED COMPONENTS
    std_msgs
    geometry_msgs
    visualization_msgs
    ros_msft_onnx_msgs
    image_transport
    roscpp
    cv_bridge
    tf
    onnxruntime_vendor
    rostest
  )
  add_rostest_gtest(${PROJECT_NAME}_testnode test/testtracker.launch test/testtracker.cpp src/ros_msft_onnx.cpp src/yolo_box.cpp  src/pose_parser.cpp)
  target_link_libraries(${PROJECT_NAME}_testnode ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} windowsapp)
endif()
