# cmake版本及工程名
cmake_minimum_required( VERSION 3.8 )
project( CMakeProject1 )
 
# 设置opencv库
set(OpenCV_DIR "D:/opencv/opencv/build") 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

 
# 设置pcl库 
set(PCL_DIR "D:/PCL 1.14.1/")
find_package( PCL 1.14.1 REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
#add_definitions(-DBOOST_ALL_NO_LIB)

 
# 编译与链接
add_executable( CMakeProject1 CMakeProject1.cpp jibian.cpp "xiaozheng.cpp")
target_link_libraries( CMakeProject1 ${OpenCV_LIBS} ${PCL_LIBRARIES} )
install(TARGETS CMakeProject1 RUNTIME DESTINATION bin)