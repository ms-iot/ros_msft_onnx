// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include <onnx/onnx_tracker.h>

namespace yolo
{
    struct YoloBox
    {
    public:
        std::string label;
        float x, y, width, height, confidence;
    };

    class YoloProcessor : public OnnxProcessor 
    {
        std::string _label;
    public:
        YoloProcessor();
        virtual bool init(rclcpp::Node::SharedPtr& node);

    protected:
        std::vector<YoloBox> GetRecognizedObjects(std::vector<float> modelOutputs, float threshold = 0.3f); 
        virtual void ProcessOutput(std::vector<float> output, cv::Mat& image);

    private:
        int GetOffset(int x, int y, int channel);
        float Sigmoid(float value);
        void Softmax(std::vector<float> &values);
    };
}