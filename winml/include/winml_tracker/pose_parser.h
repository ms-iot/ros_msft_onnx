#pragma once

#include <string>
#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <tf/tf.h>

#include "winml_tracker.h"

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

    class PoseProcessor : public WinMLProcessor
    {
        static std::vector<float> _gridX;
        static std::vector<float> _gridY;
        tf::Quaternion _modelQuat;
        std::vector<double> _modelScale;
        std::vector<double> _modelRPY;

        ros::Publisher _detect_pose_pub;

    

    public:
        PoseProcessor();

        std::vector<cv::Point3d> modelBounds;
        std::string meshResource;

        virtual bool init(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate);
    private:
        void initPoseTables();

        bool GetRecognizedObjects(std::vector<float> modelOutputs, Pose& pose);
        int GetOffset(int o, int channel);
        std::vector<float> Sigmoid(const std::vector<float>& values);
        void initMarker(visualization_msgs::Marker& mark, int32_t id, int32_t type, double x, double y, double z);
    protected:
        virtual void ProcessOutput(std::vector<float> output, cv::Mat& image);
    };
}