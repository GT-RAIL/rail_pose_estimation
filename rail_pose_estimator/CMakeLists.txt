cmake_minimum_required(VERSION 2.8.3)
project(rail_pose_estimator)

find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  cv_bridge
  std_msgs
  rail_pose_estimation_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs cv_bridge rail_pose_estimation_msgs std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
