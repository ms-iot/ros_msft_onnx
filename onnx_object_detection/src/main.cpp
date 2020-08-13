// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <onnx_object_detection/yolo_processor.h>
#include <ament_index_cpp/get_resource.hpp>

using std::placeholders::_1;

YoloProcessor g_onnx;
auto g_opts = rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true);

class OnnxObjectDetection : public rclcpp::Node
{
public:
    OnnxObjectDetection()
    : Node("object_detection", g_opts),
    image_topic_("image_raw"),
    image_pub_topic_("image_debug_raw"),
    visual_marker_topic_("visual_markers"),
    label_("person"),
    confidence_threshold_(0.7f)
    {
        this->get_parameter("image_topic", image_topic_);
        this->get_parameter("image_debug_topic", image_pub_topic_);
        this->get_parameter("confidence_threshold", confidence_threshold_);
        this->get_parameter("label", label_);
        this->get_parameter("frame_id", frame_id_);
        this->get_parameter("visual_marker_topic", visual_marker_topic_);

        publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(visual_marker_topic_, 10);
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(image_pub_topic_, 10);
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic_, 10, std::bind(&OnnxObjectDetection::topic_callback, this, _1));
    }

private:
    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {
        // Convert back to an OpenCV Image
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        cv::Size s = cv_ptr->image.size();
        cv::Rect ROI((s.width - 416) / 2, (s.height - 416) / 2, 416, 416);
        cv::Mat image_resized = cv_ptr->image(ROI);

        int32_t i = 0;
        auto boxes = g_onnx.ProcessImage(image_resized, confidence_threshold_);
        for (auto &box: boxes)
        {
            if (label_ == box.label)
            {
                publisher_->publish(getMarker(box, i++));
                RCLCPP_INFO(this->get_logger(), "matched label: %s", box.label.c_str());

                cv::Scalar color(255, 255, 0);
                cv::Rect rect;
                rect.x = std::max<int>((int)box.x, 0);
                rect.y = std::max<int>((int)box.y, 0);
                rect.height = std::min<int>(416 - rect.y, (int)box.height);
                rect.width = std::min<int>(416 - rect.x, (int)box.width);
                cv::rectangle(image_resized, rect, color, 2, 8, 0);
            }
        }

        {
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image_resized).toImageMsg();
            msg->header.frame_id = frame_id_;
            image_pub_->publish(*msg);
        }
    }

    visualization_msgs::msg::Marker getMarker(
        const YoloBox &box, int32_t id) const
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = rclcpp::Time();
        marker.ns = "onnx_object_detection";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::ARROW;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.pose.position.x = box.x + box.width / 2;
        marker.pose.position.y = box.y + box.height / 2;
        marker.pose.position.z = 0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 0.0;

        marker.scale.x = 0.5;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;

        return marker;
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

    std::string image_topic_;
    float confidence_threshold_;
    std::string label_;
    std::string frame_id_;
    std::string visual_marker_topic_;
    std::string image_pub_topic_;
};

std::string GetTinyYOLOv2ModelPath()
{
    std::string content;
    std::string prefix_path;
    ament_index_cpp::get_resource("packages", "onnx_object_detection", content, &prefix_path);
    return prefix_path + "/share/onnx_object_detection/models/tinyyolov2-8.onnx";
}

int main(int argc, char * argv[])
{
  YoloInitOptions initOptions;
  initOptions.modelFullPath = GetTinyYOLOv2ModelPath();

  g_onnx.init(initOptions);

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OnnxObjectDetection>());
  rclcpp::shutdown();
  return 0;
}
