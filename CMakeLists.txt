cmake_minimum_required(VERSION 3.8)
project(fast_lio)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

ADD_COMPILE_OPTIONS(-std=c++14)
ADD_COMPILE_OPTIONS(-std=c++14)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")

  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

# ROS dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcpputils REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(dependencies
  rclcpp
  rclcpp_components
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  visualization_msgs
  pcl_ros
  pcl_conversions
  livox_ros_driver2
)

# Thirdparty libraries
find_package(Eigen3 REQUIRED)
find_package(aircraft_geometry REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

message(Eigen: ${EIGEN3_INCLUDE_DIR})
message(STATUS "PCL: ${PCL_INCLUDE_DIRS}")

set(msg_files
  "msg/Pose6D.msg"
  "msg/FrameTiming.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
)
ament_export_dependencies(rosidl_default_runtime)

add_library(flight_odometry_output STATIC src/flight_odometry.cpp)
target_include_directories(flight_odometry_output PRIVATE include ${aircraft_geometry_INCLUDE_DIRS})
target_link_libraries(flight_odometry_output Eigen3::Eigen)
ament_target_dependencies(flight_odometry_output rclcpp rcpputils nav_msgs geometry_msgs px4_msgs)

add_executable(fastlio_mapping src/laserMapping.cpp src/constraint_selection.cpp src/plane_map.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp)
target_include_directories(fastlio_mapping PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(fastlio_mapping ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} Eigen3::Eigen $<LINK_ONLY:flight_odometry_output>)
target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})

list(APPEND EOL_LIST "foxy" "galactic" "eloquent" "dashing" "crystal")

if($ENV{ROS_DISTRO} IN_LIST EOL_LIST)
  # Custommsg to support foxy & galactic
  rosidl_target_interfaces(fastlio_mapping
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
else()
  rosidl_get_typesupport_target(cpp_typesupport_target
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
  target_link_libraries(fastlio_mapping ${cpp_typesupport_target})
endif()

ament_target_dependencies(fastlio_mapping ${dependencies})

set(hku_dir "${CMAKE_CURRENT_BINARY_DIR}/hku")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS hku/prepare.sh hku/output.patch)
execute_process(COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/hku/prepare.sh" "${hku_dir}"
                RESULT_VARIABLE hku_result)
if(NOT hku_result EQUAL 0)
  message(FATAL_ERROR "Failed to prepare the pinned HKU ROS2 sources")
endif()
add_executable(fastlio_hku ${hku_dir}/src/laserMapping.cpp ${hku_dir}/src/preprocess.cpp
               ${hku_dir}/include/ikd-Tree/ikd_Tree.cpp)
target_include_directories(fastlio_hku PRIVATE ${hku_dir}/include ${CMAKE_CURRENT_SOURCE_DIR}/include
                           ${PCL_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(fastlio_hku ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} Eigen3::Eigen flight_odometry_output)
if($ENV{ROS_DISTRO} IN_LIST EOL_LIST)
  rosidl_target_interfaces(fastlio_hku ${PROJECT_NAME} "rosidl_typesupport_cpp")
else()
  target_link_libraries(fastlio_hku ${cpp_typesupport_target})
endif()
ament_target_dependencies(fastlio_hku ${dependencies})

if(BUILD_TESTING)
  enable_testing()
  add_executable(flight_residual_transport test/flight_residual_transport.cpp)
  target_include_directories(flight_residual_transport PRIVATE include)
  target_link_libraries(flight_residual_transport flight_odometry_output)
  ament_target_dependencies(flight_residual_transport rclcpp nav_msgs px4_msgs std_msgs)
  add_test(NAME flight_residual_transport COMMAND flight_residual_transport)
  set_tests_properties(flight_residual_transport PROPERTIES TIMEOUT 20
    ENVIRONMENT "ROS_DOMAIN_ID=94;ROS_LOCALHOST_ONLY=1")

  add_executable(frame_imu_test test/frame_imu_test.cpp)
  target_include_directories(frame_imu_test PRIVATE include)
  ament_target_dependencies(frame_imu_test rclcpp sensor_msgs)
  add_test(NAME frame_imu COMMAND frame_imu_test)

  add_executable(constraint_selection_test test/constraint_selection_test.cpp src/constraint_selection.cpp)
  add_executable(plane_map_test test/plane_map_test.cpp src/plane_map.cpp)
  add_executable(adaptive_scan_test test/adaptive_scan_test.cpp)
  add_executable(flight_odometry_test test/flight_odometry_test.cpp)
  foreach(regression constraint_selection_test plane_map_test adaptive_scan_test flight_odometry_test)
    target_include_directories(${regression} PRIVATE include)
    target_link_libraries(${regression} Eigen3::Eigen)
    add_test(NAME ${regression} COMMAND ${regression})
  endforeach()
endif()

# ---------------- Install --------------- #
install(TARGETS fastlio_mapping fastlio_hku
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
