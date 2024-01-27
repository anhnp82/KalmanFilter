cmake_minimum_required(VERSION 3.0)
project( KalmanFilter )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( KalmanFilter
    main.cpp
    hough_transform.cpp
    filter.cpp
    object_detection.cpp
    kalman_filter.cpp )
target_link_libraries( KalmanFilter ${OpenCV_LIBS} )