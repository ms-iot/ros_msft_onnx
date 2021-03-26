#pragma once

#include <string>
#include <vector>

#include "ros_msft_onnx.h"

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
        std::string _labelPath;
        std::vector<float> _anchors;
        std::vector<std::string> _labels;
        int _class_count;
        int _row_count;
        int _col_count;
        std::string _inputName;
        std::string _outputName;
    public:
        YoloProcessor();
        
        bool init(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate);
    protected:
        std::vector<YoloBox> GetRecognizedObjects(std::vector<float> modelOutputs, float threshold = 0.3f);
        virtual void ProcessOutput(std::vector<float> output, cv::Mat& image);
    private:
        int GetOffset(int x, int y, int channel);
        static float IntersectionOverUnion(YoloBox a, YoloBox b);
        static float Sigmoid(float value);
        static void Softmax(std::vector<float> &values);
    };
}