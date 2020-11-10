#include <gtest/gtest.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include <image_transport/image_transport.h>
#include <visualization_msgs/MarkerArray.h>

#include <vcruntime.h>
#include <windows.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.AI.MachineLearning.h>

#include <winml_tracker/winml_tracker.h>
#include <winml_tracker/pose_parser.h>

#include <string>
#include <codecvt>
#include <fstream>
#include <sstream>

using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Windows::AI::MachineLearning;
using namespace Windows::Media;
using namespace Windows::Graphics::Imaging;
using namespace Windows::Storage;
using namespace std;

using convert_type = std::codecvt_utf8<wchar_t>;
using wstring_to_utf8 = std::wstring_convert<convert_type, wchar_t>;


class MarkerHelper
{
    bool _called;
public:
    MarkerHelper()
    : _called(false)
    {
    }

    void cb(const visualization_msgs::MarkerArray::ConstPtr& msg)
    {
        _called = true;
    }

    bool wasCalled()
    {
        return _called;
    }
};

TEST(TrackerTester, poseTest)
{
    ros::NodeHandle nh;

    MarkerHelper mh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher image_pub;
    image_pub = it.advertise("debug/image", 1, true);
    ros::Subscriber sub = nh.subscribe("tracked_objects", 0, &MarkerHelper::cb, &mh);
    cv::Mat image_data = cv::imread( "C:\\ws\\eden_ws\\src\\winml_tracker\\testdata\\sample_image_1.JPG");

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_data).toImageMsg();
    EXPECT_TRUE(nullptr != msg);
    image_pub.publish(msg);
    //ros::spinOnce();

    std::vector<float> bounds 
    {
        -4.57f, -10.50f, -13.33f,
        5.54f, -10.50f, -13.33f,
        5.54f, 0.00f, -13.33f,
        -4.57f, 0.00f, -13.33f,
        -4.57f, -10.50f, 13.73f,
        5.54f, -10.50f, 13.73f,
        5.54f, 0.00f, 13.73f,
        -4.57f, 0.00f, 13.73f,
        0.48f, -5.25f, 0.20f
    };

    nh.setParam("model_bounds", bounds);
    nh.setParam("onnx_model_path", "C:\\ws\\eden_ws\\src\\winml_tracker\\testdata\\shoe.onnx");

    pose::PoseProcessor poseP;
    poseP.init(nh, nh);
    poseP.ProcessImage(msg);

    //ros::spinOnce();
    ros::spin();

    EXPECT_TRUE(mh.wasCalled());
}

int main(int argc, char** argv)
{
    init_apartment();
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "tester");
    
    int ret = RUN_ALL_TESTS();

    return ret;
}