cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(ground_based_rgbd_people_detector)
find_package(PCL 1.7 REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp)

#add_message_files(
 # FILES
  #peopledata.msg
#)

#generate_messages(
 # DEPENDENCIES
 # std_msgs
#)



catkin_package()
include_directories(${PCL_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} ${catkin_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} ${catkin_DEFINITIONS})
add_executable (ground_based_rgbd_people_detector MACOSX_BUNDLE src/main_ground_based_people_detection.cpp)
#add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS})

target_link_libraries (ground_based_rgbd_people_detector ${PCL_LIBRARIES} ${catkin_LIBRARIES})

#install(TARGETS ground_based_rgbd_people_detector
#	ARCHIEVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#	RUNTIME DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#)
