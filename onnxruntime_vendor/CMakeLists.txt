cmake_minimum_required(VERSION 3.15.3)
project(onnxruntime_vendor)

find_package(ament_cmake REQUIRED)

file(DOWNLOAD
    https://github.com/microsoft/onnxruntime/releases/download/v1.4.0/onnxruntime-win-x64-1.4.0.zip
    ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-win-x64-1.4.0.zip
    EXPECTED_HASH SHA512=fb146395efd53af7dbb659a4eef2c373d9bf1a7f59b7d73af8c07fce7082b8ba6ce9d84eac55584105352bce63fed531fbb8ab77db8dd9499e09dd48fd5dbccc
    SHOW_PROGRESS
)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime")

add_custom_target(Extract ALL
    ${CMAKE_COMMAND} -E
    tar xvzf "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-win-x64-1.4.0.zip"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime"
)

install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/onnxruntime-win-x64-1.4.0/include/
    DESTINATION include
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/onnxruntime-win-x64-1.4.0/lib/onnxruntime.dll
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/onnxruntime-win-x64-1.4.0/lib/onnxruntime.pdb
    DESTINATION bin
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/onnxruntime-win-x64-1.4.0/lib/onnxruntime.lib
    DESTINATION lib
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/onnxruntime-win-x64-1.4.0/LICENSE
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/onnxruntime-win-x64-1.4.0/ThirdPartyNotices.txt
    DESTINATION share/${PROJECT_NAME}
)

ament_package(
  CONFIG_EXTRAS "src/onnxruntime_vendor-extras.cmake"
)
