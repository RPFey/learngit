# ROS_PROGRAM

## rospy

publisher 初始化时， 设置 queue_size 为较小整数， None 表示同步通信

* [ros_numpy](https://github.com/eric-wieser/ros_numpy)

ros 中收发消息机制用的都是 byte， 采用 ros_numpy.numpify 转化为 structured numpy-array
> a demo is provided under the forlder

## roscpp

class : ros::Duration ROS 中的一个时间管理类。

### message_filter package

this package takes in different messages and send them to the callback function after a while (eg. a message synchronizer)

```c++
void callback(const ImageConstPtr& img, const CameraInfoConstPtr& cam_info) {
    // you can process camera_info and image simultaneously
}

message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "image", 1);
message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub(nh, "cam_info", 1);
TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo> sync(image_sub, info_sub, 10);
// argument 10 is the maximum number of messages the 'sync' will store.
sync.registerCallback(boost::(&callback,_1,_2));
```

相当于将两个　subscriber 捆绑成一个同时处理，并添加一个回调函数。

Time Sequence : 批处理消息

```c++
void callback(const boost::shared_ptr<M const>& );

message_filters::Subcriber<std_msgs::String> sub(nh, "topic", 1);
message_filters::TimeSequence<std_msgs::String> seq(sub, ros::Duration(0.1), ros::Duration(.01), 10);
// 等待　0.1 秒，　每0.01 秒查看消息是否超过 10 条
seq.registerCallback(callback);
```

### tf2_ros

> tf 中，得到的变换关系是从 `child_frame` --> `parent_frame`, 即

$$
    P^{parent} = T * P^{child}
$$

tf2_ros 是 ros 中一个封装的坐标转换包，用以实现 tf tree。以 amcl 中 laserscan 为例(use MessageFilter), 将得到的雷达数据转换到 odom frame 上。

```c++
tf_.reset(new tf2_ros::Buffer());
laser_scan_sub_ = new message_filters::Subscriber<sensor_msgs::LaserScan>(nh_, scan_topic_, 100);
laser_scan_filter_ =
        new tf2_ros::MessageFilter<sensor_msgs::LaserScan>(*laser_scan_sub_,
                                                            *tf_,
                                                            odom_frame_id_,
                                                            100,
                                                            nh_);
laser_scan_filter_->registerCallback(boost::bind(&AmclNode::laserReceived,
                                                   this, _1));
```

class in tf2 package:

```c++
tf2_ros::Buffer() // 作为一个存储，记录了 tf tree, 能够在frame 间进行转换。

tf2_ros::BufferInterface() // Buffer 的基类

BufferInterface::transform(in, out, frame_id) //Transform an input into the target frame. The output is preallocated by the caller.

// 在定义了　tf2_ros::Buffer　(tf_) 之后
tf2::Quaternion q;
q.setRPY(0.0, 0.0, laser_scan->angle_min);
geometry_msgs::QuaternionStamped min_q;
min_q.header.stamp = laser_scan->header.stamp;
min_q.header.frame_id = laser_scan->header.frame_id;
tf2::convert(q, min_q.quaternion);

try{
  tf_->transform(min_q, min_q, base_frame_id_); //base_frame_id_ is the target frame
}
catch (tf2::TransformException& e){
  ROS_WARN(".... %s", e.what());
}
// convert the angle to base frame

double angle_min = tf2::getYaw(min_q.quaternion);
```

functions in tf2 namespace:

toMsg() (in, out) 参数时，将tf message 转换成 point(Vector3d)　形式。

tf2::convert 支持数据格式转换(tf::transform, tf2::quanternion, posestamped)

### cv_bridge

ros 中将 Image topic 与 cv::Mat 相互转化。
>cv_bridge requires certain versions of opencv(3.x), it's better to download the source code and specify the opencv package. The confluct of image code shared lib may cause the failure of func imsave.

```c++
namespace cv_bridge {
   class CvImage
   {
   public:
     std_msgs::Header header;
     std::string encoding;
     cv::Mat image;
   };
   typedef boost::shared_ptr<CvImage> CvImagePtr;
   typedef boost::shared_ptr<CvImage const> CvImageConstPtr;
}
```

也就是将 ros 中的数据格式与 Mat 相互转化
