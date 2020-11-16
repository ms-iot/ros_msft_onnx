#include <rclcpp/rclcpp.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d/calib3d.hpp>

#include <onnx_object_detection/onnx_tracker.h>

using namespace std;

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("onnx_processor");

    OnnxTracker tracker;
    tacker.init(node);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}