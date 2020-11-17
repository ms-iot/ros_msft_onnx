# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 2.8.3)
project(onnxruntime_vendor)

find_package(catkin)

catkin_package()

option(CUDA_SUPPORT "use CUDA support onnxruntime library" ON)

if(CUDA_SUPPORT)
    set(PACKAGE_URL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu/1.4.0")
    set(PACKAGE_SHA512 "c9c2ba5c594c92c1e426e9c53f9909e8851a41c99f48f8a369e082f8047d521b236f2fbb943e73975cbb45bd9957f20139c25959e50e1679dca9eeac08f73b31")
else()
    set(PACKAGE_URL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/1.4.0")
    set(PACKAGE_SHA512 "cbff106bb1f114ee1d510b594abb487e9f965b7b7a3f37b92846013eb086126a4cd69eb4564717fe1acf04d4399d1fcd0a52c3ca508f330e91a3e5d0fe560ca3")
endif()

file(DOWNLOAD
    ${PACKAGE_URL}
    ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.nuget
    EXPECTED_HASH SHA512=${PACKAGE_SHA512}
    SHOW_PROGRESS
)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime")

add_custom_target(ExtractOnnxruntime ALL
    ${CMAKE_COMMAND} -E
    tar xvzf "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.nuget"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime"
)

install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/build/native/include/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/runtimes/win-x64/native/onnxruntime.dll
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/runtimes/win-x64/native/onnxruntime.pdb
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/runtimes/win-x64/native/onnxruntime.lib
    DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/runtimes/linux-x64/native/libonnxruntime.so
    DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RENAME libonnxruntime.so.1.4.0
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/LICENSE.txt
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/ThirdPartyNotices.txt
    DESTINATION share/${PROJECT_NAME}
)

#ament_package(
#  CONFIG_EXTRAS "src/onnxruntime_vendor-extras.cmake"
#)

###################################
## catkin specific configuration ##
###################################
catkin_package(
    CFG_EXTRAS  "onnxruntime_vendor-extras.cmake"
)
