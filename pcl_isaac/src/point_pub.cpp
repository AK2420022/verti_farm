#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <chrono>
#include <memory>
#include <string>
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

/* This example creates a subclass of Node and uses a fancy C++11 lambda
* function to shorten the callback syntax, at the expense of making the
* code somewhat more difficult to understand at first glance. */

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("minimal_publisher"), count_(0)
  {
    publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("point_cloud",10);
    auto timer_callback =
      [this]() -> void {
        pcl::PointCloud<pcl::PointXYZRGB> cloud_;
        int num_points_ = 1000;
        for (int i = 0; i < num_points_; ++i) {
        const float fr = static_cast<float>(i) / static_cast<float>(num_points_);
        pcl::PointXYZRGB pt;
        pt = pcl::PointXYZRGB(fr * 255, 255 - fr * 255, 18 + fr * 20);
        pt.x = cos(fr * M_PI * 2.0) * 1.0;
        pt.y = sin(fr * M_PI * 2.0) * 1.0;
        pt.z = 0.0;
        cloud_.points.push_back(pt);
        }
        //sensor_msgs::msg::PointCloud2 msg;
        //pcl::toROSMsg(cloud_, msg);
        sensor_msgs::msg::PointCloud2 msg;

        // Fill in the size of the cloud
        msg.height = 480;
        msg.width = 640;

        // Create the modifier to setup the fields and memory
        sensor_msgs::PointCloud2Modifier mod(msg);

        // Set the fields that our cloud will have
        mod.setPointCloud2FieldsByString(2, "xyz", "rgb");

        // Set up memory for our points
        mod.resize(msg.height * msg.width);

        // Now create iterators for fields
        sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(msg, "r");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(msg, "g");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(msg, "b");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_r, ++iter_g, ++iter_b)
        {
        *iter_x = 0.0 + 1.0;
        *iter_y = 0.0 + 2.0;
        *iter_z = 0.0 + 1.3;
        *iter_r = 0;
        *iter_g = 255;
        *iter_b = 0;
        }
        this->publisher_->publish(msg);
      };
    timer_ = this->create_wall_timer(500ms, timer_callback);
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  size_t count_;

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}


