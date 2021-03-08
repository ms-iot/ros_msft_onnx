#pragma once

#include <string>
#include <vector>

#include "ros_msft_onnx.h"

namespace yolo
{
    struct YoloInitOptions
    {
        std::string modelFullPath;
    };

    struct YoloBox
    {
    public:
        std::string label;
        float x, y, width, height, confidence;
    };

    class YoloProcessor : public OnnxProcessor
    {
        std::string _label;
        std::string _label_file_path;
        std::vector<std::string> _labels{
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        };
        int _class_count;
        std::string _trackerType;
        std::vector<float> _anchors{
            1.08f, 1.19f, 3.42f, 4.41f, 6.63f, 11.38f, 9.42f, 5.11f, 16.62f, 10.52f
        };
        int _tensor_width;
        int _tensor_height;
        int _row_count;
        int _col_count;
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