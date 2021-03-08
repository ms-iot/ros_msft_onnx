#pragma once

#include <onnxruntime_cxx_api.h>

class OnnxProcessor
{
public:
    OnnxProcessor();

    void ProcessImage(const sensor_msgs::ImageConstPtr& image);

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

    virtual bool init(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate);

private:
    ImageProcessing _process;

protected:
    virtual void ProcessOutput(std::vector<float> output, cv::Mat& image) = 0;
    bool _fake;
    uint32_t _tensorWidth;
    uint32_t _tensorHeight;
    std::shared_ptr<Ort::Env> _env;
    std::shared_ptr<Ort::Session> _session;
    std::shared_ptr<Ort::AllocatorWithDefaultOptions> _allocator;
    std::vector<const char*> _input_node_names;
    std::vector<const char*> _output_node_names;

    std::string _linkName;
    std::string _onnxModel;
    std::string _calibration;

    cv::Mat _camera_matrix;
    cv::Mat _dist_coeffs;

    float _confidence;

    bool _debug;
    bool _normalize;

    ros::Publisher _detect_pub;
    image_transport::Publisher _image_pub;
    image_transport::Publisher _debug_image_pub;
    image_transport::Subscriber _cameraSub;

};

class OnnxTracker
{
    ros::NodeHandle _nh;
    ros::NodeHandle _nhPrivate;

    std::shared_ptr<OnnxProcessor> _processor;

public:
    OnnxTracker() { };

    bool init(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate);
    bool shutdown();
};

