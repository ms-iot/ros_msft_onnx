#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <cv_bridge/cv_bridge.h>

#include <onnxruntime_cxx_api.h>

#include "onnx_msgs/msg/detected_object_pose.hpp"


class OnnxProcessor
{
public: 
    OnnxProcessor();
    void ProcessImage(const sensor_msgs::msg::Image::SharedPtr msg);
    virtual bool init(rclcpp::Node::SharedPtr& node); 
    void DumpParameters();

    typedef enum 
    {
        Scale,
        Crop,
        Resize
    } ImageProcessing;

    void setImageProcessing(ImageProcessing process)
    {
        _process = process;
    }

private:
    ImageProcessing _process;

protected:
    virtual void ProcessOutput(std::vector<float> output, cv::Mat& image) = 0;
    rclcpp::Node::SharedPtr _node;

    bool _fake;
    std::vector<const char*> _inName;
    std::vector<const char*> _outName;
    std::string _linkName; 
    std::string _onnxModel; 

    std::string _calibration; 
    cv::Mat _camera_matrix;
    cv::Mat _dist_coeffs;

    float _confidence;

    bool _debug;
    bool _normalize;

    uint _tensorWidth;
    uint _tensorHeight;

    int _channelCount;
    int _rowCount;
    int _colCount;

    // Session params for onnx runtime
    std::shared_ptr<Ort::Env> _env;
    std::shared_ptr<Ort::Session> _session;
    std::shared_ptr<Ort::AllocatorWithDefaultOptions> _allocator;
    std::vector<const char*> _input_node_names;
    std::vector<const char*> _output_node_names;
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_; 
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr publisher_; 
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_; 
    rclcpp::Publisher<onnx_msgs::msg::DetectedObjectPose>::SharedPtr detect_pose_pub_; 
};

class OnnxTracker
{
    rclcpp::Node::SharedPtr node;

    std::shared_ptr<OnnxProcessor> _processor;

public: 
    OnnxTracker() { };
    bool init(rclcpp::Node::SharedPtr& node);
};