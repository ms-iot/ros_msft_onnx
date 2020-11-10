#pragma once

#include <string>
#include <vector>

#include "winml_tracker.h"

namespace yolo
{
    struct YoloBox
    {
    public:
        std::string label;
        float x, y, width, height, confidence;
    };

    class YoloProcessor : public WinMLProcessor
    {
        std::string _label;
    public:
        YoloProcessor();
        
        bool init(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate);
    protected:
        std::vector<YoloBox> GetRecognizedObjects(std::vector<float> modelOutputs, float threshold = 0.3f);
        virtual void ProcessOutput(std::vector<float> output, cv::Mat& image);
    private:
        static int GetOffset(int x, int y, int channel);
        static float IntersectionOverUnion(YoloBox a, YoloBox b);
        static float Sigmoid(float value);
        static void Softmax(std::vector<float> &values);
    };
}