cmake_minimum_required(VERSION 3.12)

project(masks2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(build_static_lib true) # for building and linking easyloggingpp
set(CMAKE_PROJECT_NAME darkutils)

include(FetchContent)
FIND_PACKAGE ( OpenCV CONFIG REQUIRED )

add_definitions(-DELPP_THREAD_SAFE -DELPP_FORCE_USE_STD_THREAD)

FetchContent_Declare(easyloggingpp
    GIT_REPOSITORY https://github.com/amrayn/easyloggingpp.git
    GIT_TAG        v9.96.7
)
FetchContent_MakeAvailable(easyloggingpp)
include_directories(${easyloggingpp_SOURCE_DIR}/src)

FetchContent_Declare(ini_parser
GIT_REPOSITORY https://github.com/Roman-/ini_file_parser.git
GIT_TAG        2.1
)
FetchContent_MakeAvailable(ini_parser)
include_directories(${ini_parser_SOURCE_DIR})

add_executable(darkutils
    src/main.cpp
    src/dumanager.cpp
    )

target_include_directories(darkutils PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src" ${OpenCV_INCLUDE_DIRS})

target_link_libraries(darkutils PUBLIC ${OpenCV_LIBS} easyloggingpp -lpthread libdarkhelp.so libdarknet.so)
