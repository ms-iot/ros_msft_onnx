cmake_minimum_required(VERSION 2.8.3)
project(ros_msft_onnx)
#set(CMAKE_BUILD_TYPE Release)
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
  roscpp
  cv_bridge
  tf
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS std_msgs geometry_msgs visualization_msgs ros_msft_onnx_msgs image_transport roscpp cv_bridge tf
  CFG_EXTRAS  "onnxruntime_vendor-extras.cmake"
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/ros_msft_onnx.cpp src/main.cpp src/yolo_box.cpp src/pose_parser.cpp)
add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${EIGEN3_LIBS})

message("Installing onnxruntime_vendor Nuget package")

if(CUDA_SUPPORT)
  set(PACKAGES_CONFIG "packages.gpu.config")
  set(ONNX_RUNTIME "Microsoft.ML.OnnxRuntime.Gpu.1.4.0")
else()
  set(PACKAGES_CONFIG "packages.cpu.config")
  set(ONNX_RUNTIME "Microsoft.ML.OnnxRuntime.1.4.0")
endif()

if(MSVC)
  set(ARCH "win-x64" )
  find_program(NUGET nuget PATHS C:\\opt\\ros\\noetic\\x64\\bin)

  if(NOT NUGET)
    message(FATAL "CMake could not find the nuget command line tool. Please install it from nuget.org")
  else()
    execute_process(COMMAND ${NUGET} restore ${CMAKE_CURRENT_SOURCE_DIR}/src/${PACKAGES_CONFIG} -PackagesDirectory ${CMAKE_CURRENT_BINARY_DIR})
  endif()

else()
  set(ARCH "linux-x64")
  find_program(MONO mono PATHS /usr/bin)
  if(NOT MONO)
    message(FATAL "CMake could not find mono which is required for running nuget command line tool. Please follow the instruction from https://docs.microsoft.com/en-us/nuget/install-nuget-client-tools")
  else()
    execute_process(COMMAND ${MONO} /usr/local/bin/nuget.exe restore ${CMAKE_CURRENT_SOURCE_DIR}/src/${PACKAGES_CONFIG} -PackagesDirectory ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endif()


include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}/${ONNX_RUNTIME}/build/native/include/
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/${PACKAGES_CONFIG} ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGES_CONFIG} COPYONLY)

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
