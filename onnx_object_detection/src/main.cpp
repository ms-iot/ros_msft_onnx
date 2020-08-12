#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <onnx_object_detection/yolo_processor.h>
#include <ament_index_cpp/get_resource.hpp>

using std::placeholders::_1;

YoloProcessor g_onnx;

class OnnxObjectDetection : public rclcpp::Node
{
  public:
    OnnxObjectDetection()
    : Node("object_detection"),
    image_topic_("/image_raw")
    {
      this->get_parameter("image_topic", image_topic_);
      subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, 10, std::bind(&OnnxObjectDetection::topic_callback, this, _1));
    }

  private:
    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {
      g_onnx.ProcessImage(msg);
    }
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;

    std::string image_topic_;
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
