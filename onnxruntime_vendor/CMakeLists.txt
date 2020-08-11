cmake_minimum_required(VERSION 3.15.3)
project(onnxruntime_vendor)

find_package(ament_cmake REQUIRED)

find_program(NUGET nuget)

file(DOWNLOAD
    https://dist.nuget.org/win-x86-commandline/latest/nuget.exe
    ${CMAKE_CURRENT_BINARY_DIR}/nuget.exe
    EXPECTED_HASH SHA512=a4be44f8a9f4868a91c1f0336fae598b4d3dc3f1eecac1a8e0c9dd81d8adc100ca127b40449b01d0ad7f52074d3c83a9d19d5356a776f95f88f41d1f0b297bf5
    SHOW_PROGRESS
)

add_custom_target(NugetRestore ALL ${CMAKE_CURRENT_BINARY_DIR}\\nuget.exe restore ${CMAKE_CURRENT_LIST_DIR}\\src\\packages.config -SolutionDirectory ${CMAKE_CURRENT_BINARY_DIR} -ConfigFile ${CMAKE_CURRENT_LIST_DIR}\\src\\nuget.config)

install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/packages/Microsoft.ML.OnnxRuntime.1.4.0/build/native/include/
    DESTINATION include
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/packages/Microsoft.ML.OnnxRuntime.1.4.0/runtimes/win-x64/native/onnxruntime.dll
        ${CMAKE_CURRENT_BINARY_DIR}/packages/Microsoft.ML.OnnxRuntime.1.4.0/runtimes/win-x64/native/onnxruntime.pdb
    DESTINATION bin
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/packages/Microsoft.ML.OnnxRuntime.1.4.0/runtimes/win-x64/native/onnxruntime.lib
    DESTINATION lib
)

ament_package(
  CONFIG_EXTRAS "src/onnxruntime_vendor-extras.cmake"
)
