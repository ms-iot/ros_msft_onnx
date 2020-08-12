#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <object_tracker/yolo_processor.h>

using std::placeholders::_1;

YoloProcessor g_onnx;

class MinimalSubscriber : public rclcpp::Node
{
  public:
    MinimalSubscriber()
    : Node("minimal_subscriber")
    {
      std::string default_image = "/image_raw";
      this->get_parameter_or("image_topic", image_topic_, default_image);
      subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
    }

  private:
    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {
      g_onnx.ProcessImage(msg);
      //RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
    }
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;

    std::string image_topic_;
};

int main(int argc, char * argv[])
{
  g_onnx.init();

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}
