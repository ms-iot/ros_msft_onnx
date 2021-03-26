cmake_minimum_required(VERSION 3.5)
project(ros_msft_onnx)
set(CMAKE_BUILD_TYPE Relwithdebinfo)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(MSVC)

  add_compile_options(/Od)
  add_compile_options(/wd4267)
  add_compile_options(/await)

  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

endif()

option(CUDA_SUPPORT "use CUDA support onnxruntime library" OFF)

find_package(Eigen3 REQUIRED)

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  visualization_msgs
  ros_msft_onnx_msgs
  image_transport
  dynamic_reconfigure
  roscpp
  cv_bridge
  tf
)

generate_dynamic_reconfigure_options(
  cfg/reconfig.cfg
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS std_msgs geometry_msgs visualization_msgs ros_msft_onnx_msgs image_transport dynamic_reconfigure roscpp cv_bridge tf
  CFG_EXTRAS  "onnxruntime_vendor-extras.cmake"
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/ros_msft_onnx.cpp src/main.cpp src/yolo_box.cpp src/pose_parser.cpp)
add_dependencies(${PROJECT_NAME}_node ${PROJECT_NAME}_gencfg ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${EIGEN3_LIBS})

message("Installing onnxruntime_vendor Nuget package")

if(CUDA_SUPPORT)
  set(ONNX_RUNTIME "Microsoft.ML.OnnxRuntime.Gpu.1.4.0")
  set(PACKAGE_URL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu/1.4.0")
  set(PACKAGE_SHA512 "c9c2ba5c594c92c1e426e9c53f9909e8851a41c99f48f8a369e082f8047d521b236f2fbb943e73975cbb45bd9957f20139c25959e50e1679dca9eeac08f73b31")
else()
  set(ONNX_RUNTIME "Microsoft.ML.OnnxRuntime.1.4.0")
  set(PACKAGE_URL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/1.4.0")
  set(PACKAGE_SHA512 "cbff106bb1f114ee1d510b594abb487e9f965b7b7a3f37b92846013eb086126a4cd69eb4564717fe1acf04d4399d1fcd0a52c3ca508f330e91a3e5d0fe560ca3")
endif()

file(DOWNLOAD
    ${PACKAGE_URL}
    ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.nuget
    EXPECTED_HASH SHA512=${PACKAGE_SHA512}
    SHOW_PROGRESS
)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${ONNX_RUNTIME}")

if(MSVC)
  set(ARCH "win-x64" )
  execute_process(COMMAND tar xvzf "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.nuget" 
               WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${ONNX_RUNTIME}"
               )

else()
  set(ARCH "linux-x64")
  execute_process(COMMAND unzip "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.nuget" 
               WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${ONNX_RUNTIME}"
               )

endif()

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}/${ONNX_RUNTIME}/build/native/include/
)

if(MSVC)
  target_link_libraries(${PROJECT_NAME}_node ${CMAKE_CURRENT_BINARY_DIR}/${ONNX_RUNTIME}/runtimes/${ARCH}/native/onnxruntime.lib)
  configure_file(${CMAKE_CURRENT_BINARY_DIR}/${ONNX_RUNTIME}/runtimes/${ARCH}/native/onnxruntime.dll ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/onnxruntime.dll COPYONLY)
  configure_file(${CMAKE_CURRENT_BINARY_DIR}/${ONNX_RUNTIME}/runtimes/${ARCH}/native/onnxruntime.lib ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/onnxruntime.lib COPYONLY)
  configure_file(${CMAKE_CURRENT_BINARY_DIR}/${ONNX_RUNTIME}/runtimes/${ARCH}/native/onnxruntime.pdb ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/onnxruntime.pdb COPYONLY)
else()
  target_link_libraries(${PROJECT_NAME}_node ${CMAKE_CURRENT_BINARY_DIR}/${ONNX_RUNTIME}/runtimes/${ARCH}/native/libonnxruntime.so)
  configure_file(${CMAKE_CURRENT_BINARY_DIR}/${ONNX_RUNTIME}/runtimes/${ARCH}/native/libonnxruntime.so ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libonnxruntime.so COPYONLY)
endif()

# The node expects to use the Tiny YOLO model available in the ONNX model zoo.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/testdata/model.onnx")
file(DOWNLOAD
  https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx
  ${CMAKE_CURRENT_SOURCE_DIR}/testdata/model.onnx
  SHOW_PROGRESS
)
endif()
