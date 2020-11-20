#include <rclcpp/rclcpp.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d/calib3d.hpp>

#include <onnx/onnx_tracker.h>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("onnx_processor");

    OnnxTracker tracker;
    
    if(tracker.init(node))
    {
        rclcpp::spin(node);
        RCLCPP_INFO(node->get_logger(), "Shutting down onnx_processor node");
        rclcpp::shutdown();
        return 0;
    }
    else 
    {
        return 1;
    }

}