#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <opencv2/calib3d/calib3d.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <onnx/onnx_tracker.h>
#include <onnx/pose_processor.h>
#include "onnx_msgs/msg/detected_object_pose.hpp"

#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(10)
#include <Eigen/Eigen>

#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <codecvt>
#include <fstream>
#include <sstream>

const int IMAGE_WIDTH = 416;
const int IMAGE_HEIGHT = 416;

const int ROW_COUNT = 13;
const int COL_COUNT = 13;
const int CHANNEL_COUNT = 20;
const int CLASS_COUNT = 20;

using namespace std;
using namespace pose;

bool g_init = false;
std::vector<float> PoseProcessor::_gridX;
std::vector<float> PoseProcessor::_gridY;

PoseProcessor::PoseProcessor()
:_modelQuat(0.0, 0.0, 0.0, 0.0)
{
    _normalize = true;
}


bool PoseProcessor::init(rclcpp::Node::SharedPtr& node) 
{
    initPoseTables();

    OnnxProcessor::init(node);
    _channelCount = CHANNEL_COUNT;
    _rowCount = ROW_COUNT;
    _colCount = COL_COUNT;
    _outName = {"218"};
    _inName = {"0"};

    if (!_node->get_parameter("mesh_rotation", _modelRPY) ||
        _modelRPY.size() != 3)
    {
        _modelRPY.push_back(0.0f);
        _modelRPY.push_back(0.0f);
        _modelRPY.push_back(0.0f);
    }

    if (!_node->get_parameter("mesh_scale", _modelScale) ||
        _modelScale.size() != 3)
    {
        _modelScale.push_back(1.0f);
        _modelScale.push_back(1.0f);
        _modelScale.push_back(1.0f);
    }

    _modelQuat.setRPY(_modelRPY[0], _modelRPY[1], _modelRPY[2]);
    _modelQuat.normalize();

    _node->get_parameter("mesh_resource", meshResource);

    std::vector<double> points;
    if (_node->get_parameter("model_bounds", points))
    {
        if (points.size() < 9 * 3)
        {
            RCLCPP_ERROR(_node->get_logger(), "Model Bounds needs 9 3D floating points.");
            return false; 
        }

        for (int p = 0; p < points.size(); p += 3)
        {
            modelBounds.push_back(cv::Point3d(points[p], points[p + 1], points[p + 2]));
        }

        return true;
    }
    else
    {
        RCLCPP_ERROR(_node->get_logger(), "Model Bounds needs to be specified for Pose processing.");
        return false;
    }
}


void PoseProcessor::initPoseTables()
{
    if (g_init)
    {
        return;
    }
    else
    {
        g_init = true;

        int xCount = 0;
        int yCount = 0;
        float yVal = 0.0f;

        for (int y = 0; y < ROW_COUNT; y++)
        {
            for (int x = 0; x <= COL_COUNT; x++) // confirm <= 
            {
                _gridX.push_back((float)xCount);
                _gridY.push_back(yVal);

                if (yCount++ == COL_COUNT - 1) // confirm col - 1
                {
                    yVal += 1.0;
                    yCount = 0;
                }

                if (xCount == COL_COUNT - 1)
                {
                    xCount = 0;
                }
                else
                {
                    xCount++;
                }
            }
        }
    }
}

std::vector<float> operator+(const std::vector<float>& a, const std::vector<float>& b)
{
    std::vector<float> ret;
    std::vector<float>::const_iterator aptr = a.begin();
    std::vector<float>::const_iterator bptr = b.begin();
    for (; 
        aptr < a.end() && bptr < b.end(); 
        aptr++, bptr++)
    {
        ret.push_back(*aptr + *bptr);
    }

    return ret;
}

void PoseProcessor::initMarker(visualization_msgs::msg::Marker& marker, int32_t id, int32_t type, double x, double y, double z)
{
    marker.header.frame_id = _linkName;
    marker.header.stamp = rclcpp::Time();
    marker.ns = "onnx_pose_detection";
    marker.id = id;
    marker.type = type;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.mesh_use_embedded_materials = true;
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = z;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.points.clear();
    
    if (type == visualization_msgs::msg::Marker::ARROW)
    {
        marker.scale.x = 0.01;
        marker.scale.y = 0.012;
        marker.scale.z = 0.0;
    }
    else if (type == visualization_msgs::msg::Marker::SPHERE)
    {
        marker.scale.x = 0.01;
        marker.scale.y = 0.01;
        marker.scale.z = 0.01;
    }
    else
    {
        marker.scale.x = _modelScale[0];
        marker.scale.y = _modelScale[1];
        marker.scale.z = _modelScale[2];
    }

    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
}


void PoseProcessor::ProcessOutput(std::vector<float> output, cv::Mat& image)
{

    if (_fake)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = _linkName;
        marker.header.stamp = rclcpp::Time();
        marker.ns = "onnx_pose_detection";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.mesh_resource = meshResource;
        marker.mesh_use_embedded_materials = true;

        marker.pose.position.x = 0.30f;
        marker.pose.position.y = 0.0f;
        marker.pose.position.z = 0.0f;

        tf2::Quaternion modelQuat(0.0,0.0,0.0,1.0);

        modelQuat = _modelQuat * modelQuat;
        modelQuat.normalize();

        // tf::quaternionTFToMsg(modelQuat, marker.pose.orientation);
        auto quat_msg = tf2::toMsg(modelQuat);
        marker.pose.orientation = quat_msg;

        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;

        publisher_->publish(marker);
        return;
    }

    std::vector<int> cuboid_edges_v1({1, 2, 3, 4, 5, 6, 7, 8, 2, 1, 3, 4});
    std::vector<int> cuboid_edges_v2({2, 3, 4, 1, 6, 7, 8, 5, 6, 5, 7, 8});

    Pose pose;
    if (GetRecognizedObjects(output, pose))
    {
        if (_calibration.empty())
        {
            // Borrowing from https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
            double focal_length = 416; // Approximate focal length.
            cv::Point2d center = cv::Point2d(focal_length / 2, focal_length / 2);
            _camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
            _dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion
        }

        cv::Mat rvec(3, 1, cv::DataType<double>::type);
        cv::Mat tvec(3, 1, cv::DataType<double>::type);

        // Solve for pose
        if (cv::solvePnP(modelBounds, pose.bounds, _camera_matrix, _dist_coeffs, rvec, tvec))
        {
            std::vector<cv::Point3f> AxisPoints3D;
            AxisPoints3D.push_back(cv::Point3f( 0,  0,  0));
            AxisPoints3D.push_back(cv::Point3f(20,  0,  0));
            AxisPoints3D.push_back(cv::Point3f( 0, 20,  0));
            AxisPoints3D.push_back(cv::Point3f( 0,  0, 20));

            std::vector<cv::Point2f> AxisPoints2D;
            cv::projectPoints(AxisPoints3D, rvec, tvec, _camera_matrix, _dist_coeffs, AxisPoints2D);

            cv::line(image, AxisPoints2D[0], AxisPoints2D[1], cv::Scalar(255, 0, 0), 2);
            cv::line(image, AxisPoints2D[0], AxisPoints2D[2], cv::Scalar(0, 255, 0), 2);
            cv::line(image, AxisPoints2D[0], AxisPoints2D[3], cv::Scalar(0, 0, 255), 2);

            cv::Mat_<double> rvecRod(3, 3);
            cv::Rodrigues(rvec, rvecRod);

            tf2::Matrix3x3 tfRod(
                rvecRod(0, 0), rvecRod(0, 1), rvecRod(0, 2),
                rvecRod(1, 0), rvecRod(1, 1), rvecRod(1, 2),
                rvecRod(2, 0), rvecRod(2, 1), rvecRod(2, 2)
                );
            tf2::Quaternion poseQuat;
            tfRod.getRotation(poseQuat);

            // std::vector<visualization_msgs::msg::Marker> markers;
            visualization_msgs::msg::Marker marker;
            double x = tvec.at<double>(0) / 1000.0;
            double y = tvec.at<double>(1) / 1000.0;
            double z = tvec.at<double>(2) / 1000.0;

            initMarker(marker, 0, visualization_msgs::msg::Marker::SPHERE, x, y, z);
            marker.mesh_resource = meshResource;

            // tf::quaternionTFToMsg(poseQuat, marker.pose.orientation);
            auto quat_msg = tf2::toMsg(poseQuat);
            marker.pose.orientation = quat_msg;
            publisher_->publish(marker);
            // markers.push_back(marker);

            onnx_msgs::msg::DetectedObjectPose doPose; 

            doPose.header.frame_id = _linkName;
            doPose.header.stamp = rclcpp::Time();
            // tf::quaternionTFToMsg(poseQuat, doPose.pose.orientation);
            quat_msg = tf2::toMsg(poseQuat);
            doPose.pose.orientation = quat_msg;
            doPose.pose.position.x = x;
            doPose.pose.position.y = y;
            doPose.pose.position.z = z;
            doPose.confidence = pose.confidence;

            for (int i = 0; i < doPose.flatbounds.size(); i++)
            {
                doPose.flatbounds[i].x = pose.bounds[i].x;
                doPose.flatbounds[i].y = pose.bounds[i].y;
                doPose.flatbounds[i].z = 0;
            }

            detect_pose_pub_->publish(doPose);

            if (_debug)
            {
                visualization_msgs::msg::Marker marker1;
                initMarker(marker1, 1, visualization_msgs::msg::Marker::ARROW, x, y, z);

                geometry_msgs::msg::Point pt;
                marker1.points.push_back(pt);
                pt.x = .1;
                marker1.points.push_back(pt);

                marker1.color.r = 1.0; marker1.color.g = 0.0; marker1.color.b = 0.0;
                // tf::quaternionTFToMsg(poseQuat, marker1.pose.orientation);
                auto quat_msg = tf2::toMsg(poseQuat);
                marker.pose.orientation = quat_msg;
                // markers.push_back(marker1);
                publisher_->publish(marker1);

                visualization_msgs::msg::Marker marker2;
                initMarker(marker2, 2, visualization_msgs::msg::Marker::ARROW, x, y, z);

                pt.x = 0.0;
                marker2.points.push_back(pt);
                pt.y = .1;
                marker2.points.push_back(pt);

                marker2.color.r = 0.0; marker2.color.g = 1.0; marker2.color.b = 0.0;
                // tf::quaternionTFToMsg(poseQuat, marker2.pose.orientation);
                quat_msg = tf2::toMsg(poseQuat);
                marker.pose.orientation = quat_msg;
                // markers.push_back(marker2);
                publisher_->publish(marker2);

                visualization_msgs::msg::Marker marker3;
                initMarker(marker3, 3, visualization_msgs::msg::Marker::ARROW, x, y, z);

                pt.x = 0.0; pt.y = 0.0;
                marker3.points.push_back(pt);
                pt.z = .1;
                marker3.points.push_back(pt);

                marker3.color.r = 0.0; marker3.color.g = 0.0; marker3.color.b = 1.0;
                // tf::quaternionTFToMsg(poseQuat, marker3.pose.orientation);
                quat_msg = tf2::toMsg(poseQuat);
                marker.pose.orientation = quat_msg;
                // markers.push_back(marker3);
                publisher_->publish(marker3);
            }
        }

        if (_debug)
        {
            std::vector<cv::Scalar> color({
                cv::Scalar(0, 0, 128),
                cv::Scalar(128, 0, 128),
                cv::Scalar(0, 128, 128),
                cv::Scalar(128, 0, 0),
                cv::Scalar(128, 128, 0),
                cv::Scalar(128, 128, 128),
                cv::Scalar(0, 128, 0),
                cv::Scalar(128, 0, 0),
                cv::Scalar(255, 0, 255),
                cv::Scalar(255, 255, 255) }
            );

            for (int i = 0; i < cuboid_edges_v2.size(); i++)
            {

                cv::Point2i pt1 = pose.bounds[cuboid_edges_v1[i]];
                cv::Point2i pt2 = pose.bounds[cuboid_edges_v2[i]];
                cv::line(image, pt1, pt2, color[i], 1);
            }
        }
    }

    // Always publish the resized image
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();
    image_pub_->publish(*msg);

}



bool PoseProcessor::GetRecognizedObjects(std::vector<float> modelOutputs, Pose& pose)
{
    initPoseTables();
    std::vector<std::vector<float>> output;
    for (int c = 0; c < CLASS_COUNT; c++)
    {
        std::vector<float> chanVec;
        for (int vec = 0; vec < ROW_COUNT * COL_COUNT; vec++)
        {
            chanVec.push_back(modelOutputs[GetOffset(vec, c)]);
        }

        output.push_back(chanVec);
    }

    auto xs0 = Sigmoid(output[0]) + _gridX;
    auto ys0 = Sigmoid(output[1]) + _gridY;
    auto xs1 = output[2] + _gridX;
    auto ys1 = output[3] + _gridY;
    auto xs2 = output[4] + _gridX;
    auto ys2 = output[5] + _gridY;
    auto xs3 = output[6] + _gridX;
    auto ys3 = output[7] + _gridY;
    auto xs4 = output[8] + _gridX;
    auto ys4 = output[9] + _gridY;
    auto xs5 = output[10] + _gridX;
    auto ys5 = output[11] + _gridY;
    auto xs6 = output[12] + _gridX;
    auto ys6 = output[13] + _gridY;
    auto xs7 = output[14] + _gridX;
    auto ys7 = output[15] + _gridY;
    auto xs8 = output[16] + _gridX;
    auto ys8 = output[17] + _gridY;
    auto det_confs = Sigmoid(output[18]);

    float max_conf = -1.0f;
    int max_ind = -1;
    for (int c = 0; c < ROW_COUNT * COL_COUNT; c++)
    {
        float conf = det_confs[c];

        if (conf > max_conf)
        {
            max_conf = conf;
            max_ind = c;
        }
    }

    pose.confidence = max_conf;

    if (max_conf > _confidence && max_ind >= 0)
    {
        pose.bounds.push_back({ (xs0[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys0[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
        pose.bounds.push_back({ (xs1[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys1[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
        pose.bounds.push_back({ (xs2[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys2[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
        pose.bounds.push_back({ (xs3[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys3[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
        pose.bounds.push_back({ (xs4[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys4[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
        pose.bounds.push_back({ (xs5[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys5[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
        pose.bounds.push_back({ (xs6[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys6[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
        pose.bounds.push_back({ (xs7[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys7[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });
        pose.bounds.push_back({ (xs8[max_ind] / (float)COL_COUNT) * (float)IMAGE_WIDTH, (ys8[max_ind] / (float)ROW_COUNT) * (float)IMAGE_HEIGHT });

        return true;
    }

    return false;
}

int PoseProcessor::GetOffset(int o, int channel)
{
    static int channelStride = ROW_COUNT * COL_COUNT;
    return (channel * channelStride) + o;
}

std::vector<float> PoseProcessor::Sigmoid(const std::vector<float>& values)
{
    std::vector<float> ret;

    for (std::vector<float>::const_iterator ptr = values.begin(); ptr < values.end(); ptr++)
    {
        float k = (float)std::exp(*ptr);
        ret.push_back(k / (1.0f + k));
    }

    return ret;
}