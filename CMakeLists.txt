cmake_minimum_required(VERSION 3.16)

project(VitalsMonitor
    VERSION 1.0.0
    DESCRIPTION "Contactless rPPG heart rate and ocular fatigue monitor"
    LANGUAGES CXX)

# ----------------------------------------------------------------------------
# Toolchain / build configuration
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    set(VITALS_WARNINGS /W4 /permissive-)
    set(VITALS_OPT /O2)
else()
    set(VITALS_WARNINGS -Wall -Wextra -Wpedantic -Wshadow)
    set(VITALS_OPT -O3)
endif()

option(VITALS_NATIVE_ARCH "Optimise for the host CPU (-march=native)" OFF)
option(VITALS_DOWNLOAD_MODEL "Download the YuNet ONNX model at configure time if missing" ON)
option(VITALS_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# ----------------------------------------------------------------------------
# Dependencies — FaceDetectorYN lives in objdetect (which itself requires dnn).
# ----------------------------------------------------------------------------
find_package(OpenCV 4.5.4 REQUIRED COMPONENTS core imgproc videoio highgui dnn objdetect)
message(STATUS "OpenCV ${OpenCV_VERSION} found at ${OpenCV_DIR}")

# ----------------------------------------------------------------------------
# YuNet model acquisition
#   * face_detection_yunet_2023mar.onnx needs OpenCV >= 4.8.0
#   * older OpenCV (4.5.4 – 4.7.x) needs face_detection_yunet_2022mar.onnx
# ----------------------------------------------------------------------------
if(OpenCV_VERSION VERSION_GREATER_EQUAL 4.8.0)
    set(_vitals_default_model "face_detection_yunet_2023mar.onnx")
else()
    set(_vitals_default_model "face_detection_yunet_2022mar.onnx")
endif()
set(VITALS_MODEL_NAME "${_vitals_default_model}" CACHE STRING "YuNet model file name")
if(OpenCV_VERSION VERSION_LESS 4.8.0)
    if(VITALS_MODEL_NAME MATCHES "2023mar")
        message(WARNING "OpenCV ${OpenCV_VERSION} cannot run ${VITALS_MODEL_NAME} (needs >= 4.8.0).")
    else()
        message(WARNING
            "OpenCV ${OpenCV_VERSION} < 4.8 uses ${VITALS_MODEL_NAME}, which has been retired from "
            "opencv_zoo's main branch. Upgrading to OpenCV >= 4.8 is strongly recommended; "
            "otherwise place the 2022mar model in ${CMAKE_CURRENT_SOURCE_DIR}/models/.")
    endif()
endif()

# opencv_zoo stores models in Git LFS: media.githubusercontent.com serves the
# real binary (raw.githubusercontent.com would only return the LFS pointer).
set(VITALS_MODEL_URL
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/${VITALS_MODEL_NAME}"
    CACHE STRING "Primary download URL of the YuNet model")
set(VITALS_MODEL_URL_FALLBACK
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/${VITALS_MODEL_NAME}")
set(VITALS_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models")
set(VITALS_MODEL_PATH "${VITALS_MODEL_DIR}/${VITALS_MODEL_NAME}")

function(vitals_fetch_model dst)
    file(MAKE_DIRECTORY "${VITALS_MODEL_DIR}")
    foreach(url IN ITEMS "${VITALS_MODEL_URL}" "${VITALS_MODEL_URL_FALLBACK}")
        message(STATUS "Downloading ${VITALS_MODEL_NAME} from ${url}")
        file(DOWNLOAD "${url}" "${dst}.part" STATUS st TIMEOUT 120 TLS_VERIFY ON)
        list(GET st 0 code)
        set(sz 0)
        if(EXISTS "${dst}.part")
            file(SIZE "${dst}.part" sz)
        endif()
        # A Git-LFS pointer or HTML error page is < 1 KB; the model is ~230 KB.
        if(code EQUAL 0 AND sz GREATER 100000)
            file(RENAME "${dst}.part" "${dst}")
            message(STATUS "Model saved to ${dst} (${sz} bytes)")
            return()
        endif()
        file(REMOVE "${dst}.part")
        message(STATUS "  failed (${st}, ${sz} bytes)")
    endforeach()
    message(WARNING
        "Could not download the YuNet model. Fetch it manually:\n"
        "  curl -L -o \"${dst}\" \"${VITALS_MODEL_URL}\"\n"
        "or pass --model=<path> at runtime.")
endfunction()

if(NOT EXISTS "${VITALS_MODEL_PATH}")
    if(VITALS_DOWNLOAD_MODEL)
        vitals_fetch_model("${VITALS_MODEL_PATH}")
    else()
        message(STATUS "YuNet model not found; download it manually:\n"
                       "  curl -L -o \"${VITALS_MODEL_PATH}\" \"${VITALS_MODEL_URL}\"")
    endif()
endif()

# ----------------------------------------------------------------------------
# Core library (all modules) + application
# ----------------------------------------------------------------------------
add_library(vitals_core STATIC
    src/FaceTracker.cpp
    src/SignalProcessor.cpp
    src/OcularAnalyzer.cpp
    src/TelemetryUI.cpp)
target_include_directories(vitals_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vitals_core PUBLIC ${OpenCV_LIBS})
target_compile_options(vitals_core PRIVATE ${VITALS_WARNINGS} $<$<CONFIG:Release>:${VITALS_OPT}>)

add_executable(vitals_monitor src/main.cpp)
target_link_libraries(vitals_monitor PRIVATE vitals_core)
target_compile_options(vitals_monitor PRIVATE ${VITALS_WARNINGS} $<$<CONFIG:Release>:${VITALS_OPT}>)
target_compile_definitions(vitals_monitor PRIVATE VITALS_DEFAULT_MODEL="${VITALS_MODEL_PATH}")

if(VITALS_NATIVE_ARCH AND NOT MSVC)
    target_compile_options(vitals_core PRIVATE -march=native)
    target_compile_options(vitals_monitor PRIVATE -march=native)
endif()
if(VITALS_WARNINGS_AS_ERRORS)
    if(MSVC)
        target_compile_options(vitals_core PRIVATE /WX)
    else()
        target_compile_options(vitals_core PRIVATE -Werror)
    endif()
endif()

# Optional: `cmake --build . --target fetch_model` re-downloads the model.
add_custom_target(fetch_model
    COMMAND ${CMAKE_COMMAND} -E make_directory "${VITALS_MODEL_DIR}"
    COMMAND ${CMAKE_COMMAND} "-DURL=${VITALS_MODEL_URL}" "-DDST=${VITALS_MODEL_PATH}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchModel.cmake"
    COMMENT "Downloading ${VITALS_MODEL_NAME}")

install(TARGETS vitals_monitor RUNTIME DESTINATION bin)

# ----------------------------------------------------------------------------
# Tests (synthetic signals, no camera required):  ctest --output-on-failure
# ----------------------------------------------------------------------------
option(VITALS_BUILD_TESTS "Build the synthetic regression tests" ON)
if(VITALS_BUILD_TESTS)
    enable_testing()
    add_executable(vitals_tests tests/test_pipeline.cpp)
    target_link_libraries(vitals_tests PRIVATE vitals_core)
    add_test(NAME vitals_pipeline COMMAND vitals_tests)
endif()
