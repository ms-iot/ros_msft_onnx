#pragma once

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>

struct YoloBox
{
public:
    std::string label;
    float x, y, width, height, confidence;
};

class YoloProcessor
{
public:
    YoloProcessor();

    void init();

    void ProcessImage(const sensor_msgs::msg::Image::SharedPtr image);

    typedef enum 
    {
        Scale,
        Crop,
        Resize
    } ImageProcessing;

    std::vector<YoloBox> GetRecognizedObjects(std::vector<float> modelOutputs, float threshold = 0.3f);
    void ProcessOutput(std::vector<float> output, cv::Mat& image);
    int GetOffset(int x, int y, int channel);
    float Sigmoid(float value);
    void Softmax(std::vector<float> &values);

private:
    ImageProcessing _process;
    uint32_t _tensorWidth;
    uint32_t _tensorHeight;
    std::shared_ptr<Ort::Env> _env;
    std::shared_ptr<Ort::Session> _session;
    std::shared_ptr<Ort::AllocatorWithDefaultOptions> _allocator;
    std::vector<const char*> _input_node_names;
    std::vector<const char*> _output_node_names;
    float _confidence;
};
