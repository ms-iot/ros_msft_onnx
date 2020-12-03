# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set(_IMPORT_PREFIX "${CMAKE_INSTALL_PREFIX}")

if(_IMPORT_PREFIX STREQUAL "/")
  set(_IMPORT_PREFIX "")
endif()

if (NOT TARGET onnxruntime-unofficial::onnxruntime)
    add_library(onnxruntime-unofficial::onnxruntime SHARED IMPORTED)
    set_property(TARGET onnxruntime-unofficial::onnxruntime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(onnxruntime-unofficial::onnxruntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include/onnxruntime_vendor"
    )
    if(WIN32)
        set_target_properties(onnxruntime-unofficial::onnxruntime PROPERTIES
            IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/onnxruntime.lib"
            IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/onnxruntime_vendor/onnxruntime.dll"
        )
    elseif(UNIX)
        set_target_properties(onnxruntime-unofficial::onnxruntime PROPERTIES
            IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libonnxruntime.so.1.4.0"
        )
    else()
        message(FATAL_ERROR "unsupported platform.")
    endif()
endif()

list(APPEND onnxruntime_vendor_TARGETS onnxruntime-unofficial::onnxruntime)
