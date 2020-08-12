cmake_minimum_required(VERSION 3.15.3)
project(onnxruntime_vendor)

find_package(ament_cmake REQUIRED)

file(DOWNLOAD
https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/1.4.0
    ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-1.4.0.nuget
    EXPECTED_HASH SHA512=cbff106bb1f114ee1d510b594abb487e9f965b7b7a3f37b92846013eb086126a4cd69eb4564717fe1acf04d4399d1fcd0a52c3ca508f330e91a3e5d0fe560ca3
    SHOW_PROGRESS
)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime")

add_custom_target(Extract ALL
    ${CMAKE_COMMAND} -E
    tar xvzf "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-1.4.0.nuget"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime"
)

install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/build/native/include/
    DESTINATION include
)

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

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/runtimes/linux-x64/native/libonnxruntime.so
    DESTINATION lib
    RENAME libonnxruntime.so.1.4.0
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/LICENSE.txt
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/ThirdPartyNotices.txt
    DESTINATION share/${PROJECT_NAME}
)

ament_package(
  CONFIG_EXTRAS "src/onnxruntime_vendor-extras.cmake"
)
