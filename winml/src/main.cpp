#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <image_transport/image_transport.h>
#include <visualization_msgs/MarkerArray.h>

#include <vcruntime.h>
#include <windows.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.AI.MachineLearning.h>

#include "winml_tracker/winml_tracker.h"

using namespace std;
using namespace winrt;

int main(int argc, char **argv)
{
    /*
    ROS_WARN("WINML: Waiting for Debugger");
    while (!IsDebuggerPresent())
    {
        Sleep(5);
    }
    */

    winrt::init_apartment();
    ros::init(argc, argv, "winml_tracker");

    ros::NodeHandle nh;
    ros::NodeHandle nhPrivate("~");

    WinMLTracker tracker;

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