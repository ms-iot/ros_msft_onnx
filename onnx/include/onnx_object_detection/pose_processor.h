#pragma once
#include <string>
#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <tf/tf.h>

#include <onnx_object_detection/onnx_tracker.h>

namespace pose
{
    const int ROW_COUNT = 13;
    const int COL_COUNT = 13;
    const int CHANNEL_COUNT = 20;
    class Pose
    {
    public:
        Pose() {}
        Pose(Pose&& p) : bounds(std::move(p.bounds)) {}
        std::vector<cv::Point2f> bounds;
        float confidence;
    };

    class PoseProcessor : public OnnxProcessor
    {
        static std::vector<float> _gridX;
        static std::vector<float> _gridY;
        tf::Quaternion _modelQuat;
        std::vector<double> _modelScale;
        std::vector<double> _modelRPY;
    
    public:
        PoseProcessor();
        std::vector<cv::Point3d> modelBounds;
        std::string meshResource;
        virtual void init();
    private:
        void initPoseTables();
        bool GetRecognizedObjects(std::vector<float> modelOutputs, Pose& pose);
        int GetOffset(int o, int channel);