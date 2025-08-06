# 设置CMake版本
cmake_minimum_required(VERSION 3.22)

# 工程名
project(GE)

# 设置编译参数
set(CMAKE_CXX_STANDARD 17)  # 设置C++标准
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -pthread -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# 设置OpenCV根路径
set(OPENCV_ROOT_DIR /usr/)
# 添加OpenCV头文件路径
include_directories(${OPENCV_ROOT_DIR}include/opencv4)

# 链接多线程库
link_libraries(pthread)

# 添加用户和库头文件
include_directories(./include)

# 查找源文件
aux_source_directory(./src SRC_LISTS)
aux_source_directory(./lib/ge GE_LIB_LISTS)

# 添加源文件进行编译
add_executable(${PROJECT_NAME} ${SRC_LISTS} ${LIB_LISTS} ${GE_LIB_LISTS})

# 寻找OpenCV库文件
file(GLOB OPENCV_LIBS ${OPENCV_ROOT_DIR}lib/libopencv_*.so)

# 链接OpenCV库
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBS})

# 寻找OpenCV库文件
file(GLOB ONNX_LIBS lib/onnx/libonnxruntime.so.1.17.1)

# 链接ONNX库
target_link_libraries(${PROJECT_NAME} ${ONNX_LIBS})