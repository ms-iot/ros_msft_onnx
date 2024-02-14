# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.5)
project(onnxruntime_vendor)

find_package(ament_cmake REQUIRED)

# disable Cuda build on github ci
if(DEFINED ENV{GITHUB_WORKSPACE})
    message("Detected Github CI, building without CUDA")
    option(CUDA_SUPPORT "use CUDA support onnxruntime library" OFF)
else()
    option(CUDA_SUPPORT "use CUDA support onnxruntime library" ON)
endif()

if(CUDA_SUPPORT)
    message("Building with Cuda support")
    set(PACKAGE_URL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu/1.17.0")
    set(PACKAGE_SHA512 "c9c2ba5c594c92c1e426e9c53f9909e8851a41c99f48f8a369e082f8047d521b236f2fbb943e73975cbb45bd9957f20139c25959e50e1679dca9eeac08f73b31")
else()
    message("Building without Cuda support")
    set(PACKAGE_URL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/1.17.0")
    set(PACKAGE_SHA512 "92d6c30ea93b1f82d6d8f9fefe9f4a2d9d958973840fd938eb4d33a52351cd7106780d69f7f21dada0946531defd02c812da6961204d6f1b78bf3e3d0cd2d066")
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
    DESTINATION include
)

if(MSVC)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/runtimes/win-x64/native/onnxruntime.dll
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/runtimes/win-x64/native/onnxruntime.pdb
    DESTINATION bin
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/runtimes/win-x64/native/onnxruntime.lib
    DESTINATION lib
)
else()

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/runtimes/linux-x64/native/libonnxruntime.so
    DESTINATION lib
    RENAME libonnxruntime.so.1.4.0
)
endif()

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/LICENSE.txt
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/ThirdPartyNotices.txt
    DESTINATION share/${PROJECT_NAME}
)

ament_package(
  CONFIG_EXTRAS "src/onnxruntime_vendor-extras.cmake"
)
