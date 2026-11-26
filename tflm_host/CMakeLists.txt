# TensorFlow Lite Micro, built for the PC with the reference kernels, plus the model check.
cmake_minimum_required(VERSION 3.20)
project(tflm_host C CXX)

set(TFLM ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite-micro)
set(DOWNLOADS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/downloads)

# ---- TFLM sources: the same lists the official Makefile uses (tools/make/sources.inc) ----
file(GLOB TFLM_CORE_SRCS
  ${TFLM}/tensorflow/lite/micro/*.cc
  ${TFLM}/tensorflow/lite/micro/arena_allocator/*.cc
  ${TFLM}/tensorflow/lite/micro/memory_planner/*.cc
  ${TFLM}/tensorflow/lite/micro/kernels/internal/*.cc
  ${TFLM}/tensorflow/lite/micro/kernels/internal/reference/*.cc)
list(FILTER TFLM_CORE_SRCS EXCLUDE REGEX "_test\\.cc$|/test_helpers\\.cc$|/test_helper_custom_ops\\.cc$")

# Kernel list: parse MICROLITE_CC_KERNEL_SRCS out of sources.inc so it stays in sync.
file(READ ${TFLM}/tensorflow/lite/micro/tools/make/sources.inc SOURCES_INC)
string(REGEX MATCH "MICROLITE_CC_KERNEL_SRCS :=[^\n]*\n(([^\n]+\n)+)" _ "${SOURCES_INC}")
string(REGEX MATCHALL "tensorflow/lite/micro/kernels/[A-Za-z0-9_/]+\\.cc" KERNEL_RELATIVE "${CMAKE_MATCH_1}")
list(TRANSFORM KERNEL_RELATIVE PREPEND ${TFLM}/ OUTPUT_VARIABLE TFLM_KERNEL_SRCS)
list(LENGTH TFLM_KERNEL_SRCS N_KERNELS)
message(STATUS "TFLM: ${N_KERNELS} kernel sources from sources.inc")

add_library(tflm STATIC ${TFLM_CORE_SRCS} ${TFLM_KERNEL_SRCS})
target_include_directories(tflm PUBLIC ${TFLM} ${DOWNLOADS} ${DOWNLOADS}/flatbuffers/include)
target_compile_definitions(tflm PUBLIC
  TFLM_REMOVE_VIRTUAL_DELETE TF_LITE_DISABLE_X86_NEON
  FLATBUFFERS_LOCALE_INDEPENDENT=0 FLATBUFFERS_PREFER_PRINTF=1 TF_LITE_USE_CTIME)
target_compile_features(tflm PUBLIC cxx_std_17)
target_compile_options(tflm PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti -fno-exceptions -fno-threadsafe-statics>
  -ffunction-sections -fdata-sections -w)

# ---- The model check ----
add_executable(tflm_check main.cc)
target_include_directories(tflm_check PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../models/cnn_none)
target_link_libraries(tflm_check PRIVATE tflm)
target_compile_options(tflm_check PRIVATE -fno-rtti -fno-exceptions)

# Float maths exactly like numpy (no fused multiply-add, SSE not x87), so the C
# pre-processing can be compared bit for bit with the Python one.
foreach(t tflm tflm_check)
  target_compile_options(${t} PRIVATE -ffp-contract=off -mfpmath=sse -msse2)
endforeach()

# ---- Debug tool: dump every layer's output for one window (compare with layer_diff.py) ----
add_executable(layer_dump layer_dump.cc)
target_include_directories(layer_dump PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../models/cnn_none)
target_link_libraries(layer_dump PRIVATE tflm)
target_compile_options(layer_dump PRIVATE -fno-rtti -fno-exceptions -ffp-contract=off -mfpmath=sse -msse2)
