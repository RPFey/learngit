#include "gazebo_plugins/gazebo_pos.h"

namespace gazebo
{
    void ModelPose::Load(physics::ModelPtr _parent, sdf::ElementPtr sdf)
    {
      std::string aim;

      if (sdf->HasElement("target")){
        aim = sdf->GetElement("target")->Get<std::string>();
      } else {
        aim = default_aim;
      }

      // load parameters
    this->robot_namespace_ = "";
    if (sdf->HasElement("robotNamespace"))
      this->robot_namespace_ = sdf->Get<std::string>("robotNamespace") + "/";

    if (!sdf->HasElement("topicName"))
    {
      ROS_INFO_NAMED("pose", "pose plugin missing <topicName>, defaults to /default_pose");
      this->topic_name_ = "/default_pose";
    }
    else
      this->topic_name_ = sdf->Get<std::string>("topicName");

    if (!sdf->HasElement("updateRate"))
    {
      ROS_DEBUG_NAMED("pose", "pose plugin missing <updateRate>, defaults to 0.0"
              " (as fast as possible)");
      this->update_rate_ = 0.0;
    }
    else
      this->update_rate_ = sdf->GetElement("updateRate")->Get<double>();

    if (!sdf->HasElement("vel_topic"))
    {
      ROS_DEBUG_NAMED("pose", "pose plugin missing <vel_topic>, defaults to /vel"
              " (as fast as possible)");
      this->vel_topic = "/vel";
    }
    else
      this->vel_topic = sdf->GetElement("vel_topic")->Get<std::string>();

    if (!sdf->HasElement("frameName"))
  {
    ROS_INFO_NAMED("pos", "punlish pos plugin missing <frameName>, defaults to base_link");
    this->frame_name_ = "base_link";
  }
  else
    this->frame_name_ = sdf->Get<std::string>("frameName");

    // Store the pointer to the model
    this->model = _parent;
    this->world = _parent->GetWorld();
    this->target = this->world->ModelByName(aim);

    this->linear = ignition::math::Vector3d(0, 0, 0);
    this->angular = ignition::math::Vector3d(0, 0, 0);
    alive_ = true;

      if (!ros::isInitialized())
    {
      ROS_FATAL_STREAM_NAMED("pose", "A ROS node for Gazebo has not been initialized, unable to load plugin. "
        << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
      return;
    }

    this->rosnode_ = new ros::NodeHandle(this->robot_namespace_);

    // publish multi queue
    this->pmq.startServiceThread();

    // if topic name specified as empty, do not publish
    if (this->topic_name_ != "")
    {
      this->pub_Queue = this->pmq.addPub<geometry_msgs::PoseStamped>();
      this->pub_ = this->rosnode_->advertise<geometry_msgs::PoseStamped>(
        this->topic_name_, 1);
    }

    gzmsg << "subscribe to topic " << this->vel_topic << "\n" ;
    ros::SubscribeOptions so =
      ros::SubscribeOptions::create<geometry_msgs::Twist>(this->vel_topic, 1,
          boost::bind(&ModelPose::cmdVelCallback, this, _1), 
          ros::VoidPtr(), &queue_);
    
    this->vel_sub_ = rosnode_->subscribe(so);
    callback_queue_thread_ =
      boost::thread(boost::bind(&ModelPose::QueueThread, this));
    // New Mechanism for Updating every World Cycle
    // Listen to the update event. This event is broadcast every
    // simulation iteration.
    this->update_connection_ = event::Events::ConnectWorldUpdateBegin(
        boost::bind(&ModelPose::UpdateChild, this));
    }

  void ModelPose::UpdateChild(){
    boost::mutex::scoped_lock lock(this->lock_);
    // gzmsg << "velocity X : "<<this->linear.X()<<" Y: "<<this->linear.Y()<<" Z: "<<this->linear.Z()<<"\n";
    this->model->SetLinearVel(this->linear);
    this->model->SetAngularVel(this->angular);

    common::Time cur_time = this->world->SimTime();

    if (this->update_rate_ > 0 &&
      (cur_time - this->last_time_).Double() < (1.0 / this->update_rate_))
    return;
    ignition::math::Pose3d model_pose = this->model->WorldPose();
    ignition::math::Pose3d target_pose = this->target->WorldPose();
    this->relative = model_pose.CoordPoseSolve(target_pose);

    this->pos_msg.header.frame_id = this->frame_name_;
    this->pos_msg.header.stamp.sec = cur_time.sec;
    this->pos_msg.header.stamp.nsec = cur_time.nsec;

    this->pos_msg.pose.position.x = this->relative.Pos().X();
    this->pos_msg.pose.position.y = this->relative.Pos().Y();
    this->pos_msg.pose.position.z = this->relative.Pos().Z();

    this->pos_msg.pose.orientation.x = this->relative.Rot().X();
    this->pos_msg.pose.orientation.y = this->relative.Rot().Y();
    this->pos_msg.pose.orientation.z = this->relative.Rot().Z();
    this->pos_msg.pose.orientation.w = this->relative.Rot().W();

    
    // publish to ros
    if (this->pub_.getNumSubscribers() > 0 && this->topic_name_ != "")
        this->pub_Queue->push(this->pos_msg, this->pub_);
    
    this->last_time_ = cur_time;
  }

  void ModelPose::cmdVelCallback(const geometry_msgs::Twist::ConstPtr& cmd_msg){
      boost::mutex::scoped_lock scoped_lock(this->lock_);
      this->linear = ignition::math::Vector3d(cmd_msg->linear.x, cmd_msg->linear.y, cmd_msg->linear.z);
      this->angular = ignition::math::Vector3d(cmd_msg->angular.x, cmd_msg->angular.y, cmd_msg->angular.z);
  }

  void ModelPose::FiniChild() {
    alive_ = false;
    queue_.clear();
    queue_.disable();
    rosnode_->shutdown();
    callback_queue_thread_.join();
  }

  void ModelPose::QueueThread()
  {
    static const double timeout = 0.01;
    while (alive_ && rosnode_->ok())
    {
      queue_.callAvailable(ros::WallDuration(timeout));
    }
  }
}