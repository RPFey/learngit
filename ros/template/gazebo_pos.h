#ifndef GAZEBO_POS_H__
#define GAZEBO_POS_H__

#include <string>

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <ros/advertise_options.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>

// gazebo stuff
#include <functional>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

#include <gazebo_plugins/PubQueue.h>

namespace gazebo
{
  std::string default_aim("BigBox");
  class ModelPose : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf);
    private: void LoadThread();
    protected: virtual void UpdateChild();
    protected: void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& cmd_msg);
    virtual void FiniChild();

    // Pointer to the model
    private: 
        physics::ModelPtr model;
        physics::WorldPtr world;
        physics::ModelPtr target;
        std::string robot_namespace_;
        std::string topic_name_;
        std::string vel_topic;
        double update_rate_;
        std::string frame_name_;
        geometry_msgs::PoseStamped pos_msg;

    // ros node handle
    private: ros::NodeHandle* rosnode_;
    private: ros::Publisher pub_;
    private: PubQueue<geometry_msgs::PoseStamped>::Ptr pub_Queue;
    private: PubMultiQueue pmq;
    private: ros::Subscriber vel_sub_;
    protected: boost::mutex lock_;

    ros::CallbackQueue queue_;
    boost::thread callback_queue_thread_;
    void QueueThread();

    private: event::ConnectionPtr update_connection_;
    private: ignition::math::Vector3d linear;
    private: ignition::math::Vector3d angular;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;

    public: ignition::math::Pose3d relative;
    private: common::Time last_time_;
    private: bool alive_;

  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(ModelPose);
}

#endif
