#include <rclcpp/rclcpp.hpp>
#include <onnx_msgs/DetectedObjectPose.h> 
#include <visualization_msgs/msg/marker.hpp>

class OnnxProcessor : public rclcpp::Node
{
public: 
    OnnxProcessor();
    void ProcessImage(const sensor_msgs::msg::Image::SharedPtr msg);
    virtual void init(rclcpp::Node::SharedPtr& node); 
    void DumpParameters();

protected:
    virtual void ProcessOutput(std::vector<float> output, cv::Mat& image) = 0;
    rclcpp::Node::SharedPtr _node;

    bool _fake; 
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
    rclcpp::Publisher<onnx_msgs::DetectedObjectPose>::SharedPtr detect_pose_pub_; 
};

class OnnxTracker
{
    rclcpp::Node::SharedPtr node;

    std::shared_ptr<OnnxProcessor> _processor;

public: 
    OnnxTracker() { };
    init(rclcpp::Node::SharedPtr& node); 
};