#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "rclcpp/rclcpp.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <memory>
#include <boost/make_shared.hpp>
#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include <pcl/visualization/cloud_viewer.h>
#include <tf2/transform_datatypes.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/convert.h>
#include <tf2/LinearMath/Quaternion.h>
#include <pcl_ros/transforms.hpp>
#include <pcl/common/common.h>
#include "tf2_ros/buffer.h"
pcl::PointCloud<pcl::PointXYZ> cloud;


class PCsubscriber : public rclcpp::Node {
public:
    PCsubscriber() : Node("PCSubscriber") {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "point_cloud", 10, std::bind(&PCsubscriber::cloud_callback, this, std::placeholders::_1));
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_; 

    /**
    * The function `cloud_callback` subscribes to a PointCloud2 message, transforms it to a different
    * frame, converts it to a PCL type, and saves it as a PCD file.
    * 
    * @param msg In the `cloud_callback` function, the `msg` parameter is of type
    * `sensor_msgs::msg::PointCloud2::SharedPtr`, which is a shared pointer to a
    * `sensor_msgs::msg::PointCloud2` message. This message typically contains a point cloud data with
    * information such as point coordinates,
    */
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
    // PCL still uses boost::shared_ptr internally
    //boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>> cloud =
        //boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBA>>();
        // Use for timing callback execution time
        auto start = std::chrono::high_resolution_clock::now();
        // Transform for pointcloud in world frame
        geometry_msgs::msg::TransformStamped stransform;
        try {
            stransform = tf_buffer_->lookupTransform("base_link", msg->header.frame_id,
                                                        tf2::TimePointZero, tf2::durationFromSec(3));
        }
        catch (const tf2::TransformException &ex) {
            
        }
        sensor_msgs::msg::PointCloud2 transformed_cloud;
        pcl_ros::transformPointCloud("base_link", stransform, *msg, transformed_cloud);

        // Convert ROS message to PCL type
        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromROSMsg(transformed_cloud, cloud);
        char filename[100];
        //pcl::fromROSMsg(*msg, *cloud);
        sprintf(filename, "/media/ashik/T7/omni/IsaacSim-nonros_workspaces/src/pcl_isaac/outputs/output%dx.pcd", std::chrono::system_clock::now());
        cout<<filename<<endl;
        pcl::io::savePCDFileASCII (filename, cloud);
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc,argv);
    rclcpp::spin(std::make_shared<PCsubscriber>());
    rclcpp::shutdown();

    return 0;
}