# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

message ("**** onnx ${CMAKE_CURRENT_LIST_FILE}")

get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
  #message(STATUS "${_variableName}=${${_variableName}}")
endforeach()


# Compute the installation prefix relative to this file.
get_filename_component(_IMPORT_PREFIX "${onnxruntime_vendor_BINARY_DIR}" PATH)

set(_IMPORT_PREFIX "${onnxruntime_vendor_BINARY_DIR}/onnxruntime")

if(_IMPORT_PREFIX STREQUAL "/")
  set(_IMPORT_PREFIX "")
endif()

message ("**** onnx ${_IMPORT_PREFIX}")

if (NOT TARGET onnxruntime-unofficial::onnxruntime)
    add_library(onnxruntime-unofficial::onnxruntime SHARED IMPORTED)
    set_property(TARGET onnxruntime-unofficial::onnxruntime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(onnxruntime-unofficial::onnxruntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/build/native/include"
    )
    if(WIN32)
        set_target_properties(onnxruntime-unofficial::onnxruntime PROPERTIES
            IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/runtimes/win-x64/native}/lib/onnxruntime.lib"
            IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtimes/win-x64/native}/bin/onnxruntime.dll"
        )
    elseif(UNIX)
        set_target_properties(onnxruntime-unofficial::onnxruntime PROPERTIES
            IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtimes/linux-x64/native/lib/libonnxruntime.so.1.4.0"
        )
    else()
        message(FATAL_ERROR "unsupported platform.")
    endif()
endif()

list(APPEND onnxruntime_vendor_TARGETS onnxruntime-unofficial::onnxruntime)
