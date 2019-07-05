cmake_minimum_required(VERSION 2.8.3)
project(support_surface_saver)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
)

