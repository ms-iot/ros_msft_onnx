#pragma once
#include <string>
#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <tf2/LinearMath/Quaternion.h>

#include <onnx/onnx_tracker.h>

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
        tf2::Quaternion _modelQuat;
        std::vector<double> _modelScale;
        std::vector<double> _modelRPY;
    
    public:
        PoseProcessor();

        std::vector<cv::Point3d> modelBounds;
        std::string meshResource;

        virtual bool init(rclcpp::Node::SharedPtr& node);
    private:
        void initPoseTables();

        bool GetRecognizedObjects(std::vector<float> modelOutputs, Pose& pose);
        int GetOffset(int o, int channel);
        std::vector<float> Sigmoid(const std::vector<float>& values);
        void initMarker(visualization_msgs::msg::Marker& mark, int32_t id, int32_t type, double x, double y, double z);
    protected:
        virtual void ProcessOutput(std::vector<float> output, cv::Mat& image);
    };
}