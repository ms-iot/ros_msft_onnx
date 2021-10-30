#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <image_transport/image_transport.h>
#include <visualization_msgs/MarkerArray.h>

#include "ros_msft_onnx/ros_msft_onnx.h"

#ifdef _WIN32
#include <objbase.h>
#endif

using namespace std;

int main(int argc, char **argv)
{
    #ifdef _WIN32
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    #endif

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

#ifdef _WIN32
    CoUninitialize();
#endif
}