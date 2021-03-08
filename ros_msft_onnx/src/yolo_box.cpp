#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <visualization_msgs/MarkerArray.h>

#include "ros_msft_onnx/ros_msft_onnx.h"
#include "ros_msft_onnx/yolo_box.h"

#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>

namespace yolo
{
    const int BOXES_PER_CELL = 5;
    const int BOX_INFO_FEATURE_COUNT = 5;
    const int CLASS_COUNT = 20;
    const float CELL_WIDTH = 32;
    const float CELL_HEIGHT = 32;
    const int TENSOR_WIDTH = 416;
    const int TENSOR_HEIGHT = 416;
    const std::string TRACKER_TYPE = "yolo";
    const std::string kDefaultLabel = "person";

    YoloProcessor::YoloProcessor()
    {
        _normalize = false;
    }

    bool YoloProcessor::init(ros::NodeHandle &nh, ros::NodeHandle &nhPrivate)
    {
        OnnxProcessor::init(nh, nhPrivate);
        nhPrivate.param("tracker_type", _trackerType, TRACKER_TYPE);
        if (_trackerType == "yolo")
        {
            _input_node_names = {"image"};
            _output_node_names = {"grid"};
            nhPrivate.param("label", _label, kDefaultLabel);
        }
        if (_trackerType == "customvision")
        {
            _anchors.clear();
            _anchors = {
                0.573f, 0.677f, 1.87f, 2.06f, 3.34f, 5.47f, 7.88f, 3.53f, 9.77f, 9.17f};
            nhPrivate.getParam("label_file_path", _label_file_path);
            std::ifstream file(_label_file_path);
            std::string line;
            _labels.clear();
            _input_node_names = {"data"};
            _output_node_names = {"model_outputs0"};
            if (file.is_open())
            {
                while (std::getline(file, line))
                {
                    _labels.push_back(line);
                }
                file.close();
            }

            else
            {
                ROS_ERROR("Unable to open the file containing labels");
                return false;
            }
        }
        _class_count = _labels.size();
        nhPrivate.param("tensor_width", _tensor_width, TENSOR_WIDTH);
        nhPrivate.param("tensor_width", _tensor_height, TENSOR_HEIGHT);
        _col_count = std::round(_tensor_width / CELL_WIDTH);
        _row_count = std::round(_tensor_height / CELL_HEIGHT);

        return true;
    }

    void YoloProcessor::ProcessOutput(std::vector<float> output, cv::Mat& image)
    {
        if (_fake)
        {
            return;
        }
        
        auto boxes = GetRecognizedObjects(output, _confidence);

        // If we found a person, send a message
        int count = 0;
        std::vector<visualization_msgs::Marker> markers;
        for (std::vector<YoloBox>::iterator it = boxes.begin(); it != boxes.end(); ++it)
        {
            for (auto label : _labels)
            {
                if (it->label == label)
                {
                    visualization_msgs::Marker marker;
                    marker.header.frame_id = _linkName;
                    marker.header.stamp = ros::Time();
                    marker.ns = "onnx";
                    marker.id = count++;
                    marker.type = visualization_msgs::Marker::SPHERE;
                    marker.action = visualization_msgs::Marker::ADD;

                    marker.pose.position.x = it->x + it->width / 2;
                    marker.pose.position.y = it->y + it->height / 2;
                    marker.pose.position.z = 0;
                    marker.pose.orientation.x = 0.0;
                    marker.pose.orientation.y = 0.0;
                    marker.pose.orientation.z = 0.0;
                    marker.pose.orientation.w = 1.0;

                    marker.scale.x = 1;
                    marker.scale.y = 0.1;
                    marker.scale.z = 0.1;
                    marker.color.a = 1.0;
                    marker.color.r = 0.0;
                    marker.color.g = 0.0;
                    marker.color.b = 1.0;

                    markers.push_back(marker);

                    if (_debug)
                    {
                        ROS_INFO("ONNX: %s detected!", label.c_str());
                        // Draw a bounding box on the CV image
                        cv::Scalar color(255, 255, 0);
                        cv::Rect box;
                        box.x = std::max<int>((int)it->x, 0);
                        box.y = std::max<int>((int)it->y, 0);
                        box.height = std::min<int>(image.rows - box.y, (int)it->height);
                        box.width = std::min<int>(image.cols - box.x, (int)it->width);
                        cv::rectangle(image, box, color, 2, 8, 0);
                    }
                }
            }
        }

        if (count > 0)
        {
            _detect_pub.publish(markers);
        }

        if (_debug)
        {
            // Always publish the resized image
            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
            _image_pub.publish(msg);
        }
    }

    std::vector<YoloBox> YoloProcessor::GetRecognizedObjects(std::vector<float> modelOutputs, float threshold)
    {
        int featuresPerBox = BOX_INFO_FEATURE_COUNT + _class_count;
        int stride = featuresPerBox * BOXES_PER_CELL;

        std::vector<YoloBox> boxes;

        for (int cy = 0; cy < _row_count; cy++)
        {
            for (int cx = 0; cx < _col_count; cx++)
            {
                for (int b = 0; b < BOXES_PER_CELL; b++)
                {
                    int channel = (b * (_class_count + BOX_INFO_FEATURE_COUNT));
                    float tx = modelOutputs[GetOffset(cx, cy, channel)];
                    float ty = modelOutputs[GetOffset(cx, cy, channel + 1)];
                    float tw = modelOutputs[GetOffset(cx, cy, channel + 2)];
                    float th = modelOutputs[GetOffset(cx, cy, channel + 3)];
                    float tc = modelOutputs[GetOffset(cx, cy, channel + 4)];

                    float x = ((float)cx + Sigmoid(tx)) * CELL_WIDTH;
                    float y = ((float)cy + Sigmoid(ty)) * CELL_HEIGHT;
                    float width = (float)exp(tw) * CELL_WIDTH * _anchors[b * 2];
                    float height = (float)exp(th) * CELL_HEIGHT * _anchors[b * 2 + 1];

                    float confidence = Sigmoid(tc);
                    if (confidence < threshold)
                        continue;

                    std::vector<float> classes(_class_count);
                    float classOffset = channel + BOX_INFO_FEATURE_COUNT;

                    for (int i = 0; i < _class_count; i++)
                        classes[i] = modelOutputs[GetOffset(cx, cy, i + classOffset)];

                    Softmax(classes);

                    // Get the index of the top score and its value
                    auto iter = std::max_element(classes.begin(), classes.end());
                    float topScore = (*iter) * confidence;
                    int topClass = std::distance(classes.begin(), iter);

                    if (topScore < threshold)
                        continue;

                    YoloBox top_box = {
                        _labels[topClass],
                        (x - width / 2),
                        (y - height / 2),
                        width,
                        height,
                        topScore};
                    boxes.push_back(top_box);
                }
            }
        }

        return boxes;
    }

    float YoloProcessor::IntersectionOverUnion(YoloBox a, YoloBox b)
    {
        int areaA = a.width * a.height;

        if (areaA <= 0)
            return 0;

        int areaB = b.width * b.height;
        if (areaB <= 0)
            return 0;

        int minX = std::max(a.x, b.x);
        int minY = std::max(a.y, b.y);
        int maxX = std::min(a.x + a.width, b.x + b.width);
        int maxY = std::min(a.y + a.height, b.x + b.width);
        int intersectionArea = std::max(maxY - minY, 0) * std::max(maxX - minX, 0);

        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    int YoloProcessor::GetOffset(int x, int y, int channel)
    {
        // YOLO outputs a tensor that has a shape of 125x13x13, which 
        // ONNX flattens into a 1D array.  To access a specific channel 
        // for a given (x,y) cell position, we need to calculate an offset
        // into the array
        int channelStride = _row_count * _col_count;
        return (channel * channelStride) + (y * _col_count) + x;
    }

    float YoloProcessor::Sigmoid(float value)
    {
        float k = (float)std::exp(value);
        return k / (1.0f + k);
    }

    void YoloProcessor::Softmax(std::vector<float> &values)
    {
        float max_val{ *std::max_element(values.begin(), values.end()) };
        std::transform(values.begin(), values.end(), values.begin(),
            [&](float x) { return std::exp(x - max_val); });

        float exptot = std::accumulate(values.begin(), values.end(), 0.0);
        std::transform(values.begin(), values.end(), values.begin(),
            [&](float x) { return (float)(x / exptot); });
    }
}