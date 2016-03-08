#include <ros/ros.h>
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Float32MultiArray.h"
#include "ros/console.h"

void DistanceRecieved(std_msgs::Float32MultiArray distance) {
	ROS_INFO_STREAM("Distance received!");
	
}
void AngleRecieved(std_msgs::Float32MultiArray angle) {
	ROS_INFO_STREAM("Angle received!");
	
}

int main(int argc, char** argv){
	ros::init(argc, argv, "topictest");
	ros::NodeHandle nh;
	ros::Subscriber distance_sub=nh.subscribe("ground_based_rgbd_people_detector/distance",1000, DistanceRecieved);
ros::Subscriber angle_sub=nh.subscribe("ground_based_rgbd_people_detector/angle",1000, AngleRecieved);
	ros::spin();
}

