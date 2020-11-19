#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <image_transport/image_transport.h>
#include <visualization_msgs/MarkerArray.h>

#include "ros_msft_onnx/ros_msft_onnx.h"

using namespace std;

int main(int argc, char **argv)
{
    /*
    ROS_WARN("ONNX: Waiting for Debugger");
    while (!IsDebuggerPresent())
    {
        Sleep(5);
    }
    */

    ros::init(argc, argv, "ros_msft_onnx");

    ros::NodeHandle nh;
    ros::NodeHandle nhPrivate("~");

    OnnxTracker tracker;

    if (tracker.init(nh, nhPrivate))
    {
        ros::spin();

        tracker.shutdown();

        return 0;
    }
    else
    {
        return 1;
    }
}