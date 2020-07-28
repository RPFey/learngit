
<!-- vim-markdown-toc GFM -->

- [GAZEBO](#gazebo)
  - [structure](#structure)
  - [sensor](#sensor)
  - [plugins](#plugins)
    - [model_plugin](#model_plugin)
      - [related API](#related-api)
    - [world_plugin](#world_plugin)
    - [sensor_plugin](#sensor_plugin)
  - [Node](#node)
  - [world](#world)
    - [DEM file](#dem-file)
  - [sensor](#sensor-1)
  - [model](#model)
    - [构建流程](#构建流程)
  - [urdf & sdf](#urdf--sdf)
  - [ignition](#ignition)
  - [Connect to ROS](#connect-to-ros)
    - [gazebo_ros plugins](#gazebo_ros-plugins)
    - [camera plugin parameters](#camera-plugin-parameters)

<!-- vim-markdown-toc -->

# GAZEBO

gazebo --verbose 会显示所有信息 / -u 进入时处于暂停

gazebo  系统文件夹下有纹理与模型，需要先 source /usr/share/gazebo/setup.bash

## structure

gazebo 包括 gzserver gzclient

environment variables:

`GAZEBO_MODEL_PATH`: colon-separated set of directories where Gazebo will search for models

`GAZEBO_RESOURCE_PATH`: colon-separated set of directories where Gazebo will search for other resources such as world and media files.

`GAZEBO_MASTER_URI`: URI of the Gazebo master. This specifies the IP and port where the server will be started and tells the clients where to connect to.

`GAZEBO_PLUGIN_PATH`: colon-separated set of directories where Gazebo will search for the plugin shared libraries at runtime.

`GAZEBO_MODEL_DATABASE_URI`: URI of the online model database where Gazebo will download models from.

整个机制与 ROS 很像， 有一个 master,name server, topic (communication)

## sensor

[noise_sensor](http://gazebosim.org/tutorials?tut=sensor_noise&cat=sensors)

add noise to sensors (lidar / imu / camera)

## plugins

Load functions create pointers and set it to sensors / models / worlds.

```shell
gzserver -s <plugin_filename>
```

**一定要记得在最后添加 GZ_REGISTER_WORLD_PLUGIN(WorldPluginTutorial)**

```c++
#include <gazebo/gazebo.hh>

namespace gazebo
{
  class WorldPluginTutorial : public WorldPlugin
  {
    public: WorldPluginTutorial() : WorldPlugin()
            {
              printf("Hello World!\n");
            }

    public: void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
            {
            }
  };
  GZ_REGISTER_WORLD_PLUGIN(WorldPluginTutorial)
  // register the plugin class (WORLD 可以替换为 GUI, SENSOR ...)
}
```

Load 中 _sdf是 导入的 sdf 文件，含有标签信息，其参数可以通过如下方式读取

```c++
if (_sdf->HasElement("jointName"))
    joint_name_ = _sdf->GetElement("jointName")->Get<std::string>();
else
  gzerr << "[gazebo_motor_model] Please specify a jointName, where the rotor is attached.\n";
```

### model_plugin

相关 API 在 gazebo physics。(注意 Model 类)

apply speed and velocity to a model

```c++
namespace gazebo
{
  class ModelPush : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf)
    {
      // Store the pointer to the model
      this->model = _parent;

      // Listen to the update event. This event is broadcast every
      // simulation iteration. 这里注意事件的形式。
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&ModelPush::OnUpdate, this));
    }

    // Called by the world update start event
    public: void OnUpdate()
    {
      // Apply a small linear velocity to the model.
      this->model->SetLinearVel(ignition::math::Vector3d(1, 0, 0));
    }

    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(ModelPush)
}
```

> 在赋予物体运动特性时，注意把 static 这个标签设置为 false !

`physics::Model_Ptr` 是指向模型的指针， `sdf::ElementPtr` 指向 sdf 文件中传入模型的参数，用来读取。

#### related API

* physics::ModelPtr model_

获得仿真世界的指针，一般在 Load 函数（模型装载时调用），用来取得世界的参数

```c++    
physics::WorldPtr world_ = model_ -> GetWorld(); 
```

获得模型位姿

```c++ 
ignition::math::Pose3d pose = model_->WorldPose();
```

获得模型关节 --> 力学模型
```c++
physics::JointPtr joint_ = model_->GetJoint(joint_name_);
```

### world_plugin

获取世界参数

重力加速度： `world_->Gravity()`

仿真时间: `world_->GetSimTime()`

### sensor_plugin

此时加载函数会成为,对应着调用的 sensor

```c++
void OpticalFlowPlugin::Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf);
```

* 相机传感器

```c++
// 首先要将传入的传感器指针转换为对应的传感器类型
sensors::CameraSensorPtr parentSensor = std::dynamic_pointer_cast<sensors::CameraSensor>(_sensor);
// CameraSensor or DethCameraSensor

// 获取 render::camera 指针
rendering::CameraPtr camera = parenSensor->GetCamera();

// 绑定更新函数
event::ConnectionPtr newFrameConnection;
newFrameConnection = camera->ConnectNewImageFrame
(
	boost::bind(&Update_func, this, _1,);			
)

// 激活相机
parentSensor -> SetActive(true);
```

camera 分为两类传感器，一是 sensor 中的，还有一个是 render 中的。 

sensor 中函数

SaveFrame : 直接保存图片

render 中函数

LastRenderWallTime  获取上一帧的时间
> 注意是以 us 计时

## Node

gazebo 内部也有通信机制，与 ROS 类似。一般在 `::Load` 中创建 `NodePtr` 相当于 `Node Handle`

```c++
transport::NodePtr node_handle_;
node_handle_ = transport::NodePtr(new transport::Node());
node_handle_->Init(namespace_);
transport::SubscriberPtr command_sub_ = node_handle_->Subscribe<mav_msgs::msgs::CommandMotorSpeed>("~/" + model_->GetName() + command_sub_topic_, &GazeboMotorModel::VelocityCallback, this); // 注意类中最后的 this
```

## world

### DEM file

三维地形图

## sensor

其实 camera 是 model 下 sensor 一个属性

camera 下可以自动保存图片

```xml
<model name='camera'>
      <static>true</static>
      <pose>-1 0 2 0 1 0</pose>
      <link name='link'>
        <visual name='visual'>
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
        </visual>
        <sensor name='my_camera' type='camera'>
          <camera>
            <!- set save and save path ->
            <save enabled="true">
              <path>/tmp/camera_save_tutorial</path>
            </save>
            <horizontal_fov>1.047</horizontal_fov>
            <image>
              <width>1920</width>
              <height>1080</height>
            </image>
            <clip>
              <near>0.1</near>
              <far>100</far>
            </clip>
          </camera>
          <always_on>1</always_on>
          <update_rate>30</update_rate>
        </sensor>
      </link>
    </model>
```

## model

Links : A link contains physical property of a model. Each link contain many collisions and visual elements. Try to reduce the number of links and substitute it with collision parts

visual : the visual part of the model .

collision: use simpler collision model to reduce computation time ("hit box")

Inertial : Inertial element describes mass and rotational inertia matrix.

Sensor /Light

Joints: A joint connect two links (A parent and child relation is established) 在PX4中 Joint 可以用来连接两个引入的模型。

add mesh in geometry

```xml
<visual name="visual">
  <geometry>
    <mesh><uri>model://tree/mesh/tree.dae</uri></mesh>
  </geometry>
</visual>
```

### 构建流程

1. 在文件夹下创建[name].sdf(细节描述) 与 model.config(大概描述) 文件 (注意GAZEBO_MODEL_PATH 下要求每个文件都有 model.config). model.config 中 sdf tag 中指定了 [name].sdf

2. 添加标签描述

   > pose 的坐标系默认为地系，可在下面建立 frame 指明。x,y,z,roll,yaw,pitch

添加模型中其余模型(注意：引用的model模型是由此模型的文件夹名字确定的)：

```xml
<!-- insert another model in a model -->
<include>
           <uri>model://hokuyo</uri>
           <!-- pose of this model -->
           <pose>.2 0 .2 0 0 0</pose>
      </include>
      <!-- joint connects and fixes this part-->
      <joint name = "hokuyo_joint" type="fixed">
           <child>hokuyo::link</child>
           <parent>chassis</parent>
      </joint>
```

## urdf & sdf

[sdf_reference](http://gazebosim.org/sdf)

具体文件在 robot_sim_demo 下的　urdf/ *.urdf.xacro 中，　可以看到各个　frame　之间的转换

udrf  描述机器人: 多用在ROS 下，需要将其修改才能在 gazbo 中使用。[urdf_reference](https://wiki.ros.org/urdf)

```xml
<!-- xacro 中的 "函数调用"-->
<xacro:macro name="default_link" params="prefix">
    <link name="${prefix}_link1" />
</xacro:macro>
<xacro:default_link prefix="my" />
```

文件基本构架：

以 px4 为例，定义不同的飞机组件： iris\ rplidar\ lidar

再在一台具体飞机下确定使用哪些组件

以 iris_fpv_cam 为例

```xml
<?xml version='1.0'?>
<sdf version='1.5'>
  <model name='iris_fpv_cam'>

    <include>
      <uri>model://iris</uri>
      <!-- add the iris plane -->
    </include>

    <include>
      <uri>model://fpv_cam</uri>
      <!-- add the fpv_cam component, add define the joint of this component-->
      <pose>0 0 0 0 0 0</pose>
    </include>
    <joint name="fpv_cam_joint" type="fixed">
      <child>fpv_cam::link</child>
      <parent>iris::base_link</parent>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <upper>0</upper>
          <lower>0</lower>
        </limit>
      </axis>
    </joint>

  </model>
</sdf>
```

## ignition

ignition::math 中提供了数学库

* ignition::math::Pose3d 

d 也可以是 f(float), i(integer)

获取位置和转角 : .Pos() 获取位置， .Rot() 获取旋转的四元数

位置变换: .CoorPositionAdd() 是加上一个平移， .CoorRotationAdd() 是加上一个旋转。 .CoorPositionSolve() 用来求相对位姿。

## Connect to ROS

`.launch` file to spawn the world. `.world` file is found under the `GAZEBO_RESOURCE_PATH/worlds` directory.

```xml
<launch>
  <!-- We resume the logic in empty_world.launch, changing only the name of the world to be launched -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="worlds/mud.world"/> <!-- Note: the world_name is with respect to GAZEBO_RESOURCE_PATH environmental variable -->
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
    <arg name="recording" value="false"/>
    <arg name="debug" value="false"/>
  </include>
</launch>
```

For urdf models, there are two ways to spawn.

1. ROS service call spawn method. You have to use a small python script to call the ros service
2. Model Database Method.

for `urdf` file, add following commands in the `.launch` file
```xml
<!-- Spawn a robot into Gazebo -->
<node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model" args="-file $(find baxter_description)/urdf/baxter.urdf -urdf -z 1 -model baxter" />
```

if it's a `xacro` file, you need to change the `xacro` format to `urdf` format, using the pr2 package

```bash
sudo apt-get install ros-melodic-pr2-common
```

```xml
<!-- Convert an xacro and put on parameter server -->
<param name="robot_description" command="$(find xacro)/xacro.py $(find pr2_description)/robots/pr2.urdf.xacro" />

<!-- Spawn a robot into Gazebo -->
<node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model" args="-param robot_description -urdf -model pr2" />
```

### gazebo_ros plugins

在与ROS 通信时，需要加入特定的插件，才能在 ROS 中收到相关信息。[link](http://gazebosim.org/tutorials?tut=ros_gzplugins&cat=connect_ros)
> 相应的插件源文件在[此处](https://github.com/ros-simulation/gazebo_ros_pkgs)

首先是在 Load 中导入 sdf 文件的参数，并且初始化 node 节点

```c++
if (!ros::isInitialized())
  {
    ROS_FATAL_STREAM_NAMED("planar_move", "PlanarMovePlugin (ns = " << robot_namespace_
      << "). A ROS node for Gazebo has not been initialized, "
      << "unable to load plugin. Load the Gazebo system plugin "
      << "'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
    return;
  }
rosnode_.reset(new ros::NodeHandle(robot_namespace_));
```

设置 Subscriber 与 Publisher

```c++
ros::SubscribeOptions so =
  ros::SubscribeOptions::create<geometry_msgs::Twist>(command_topic_, 1,
      boost::bind(&GazeboRosPlanarMove::cmdVelCallback, this, _1), 
      ros::VoidPtr(), &queue_);
\\ topic name, call back funtion, .., buffer queue
vel_sub_ = rosnode_->subscribe(so);
```

### camera plugin parameters

针对相机的：

```xml
<plugin name="camera_controller" filename="libgazebo_ros_camera.so">
                <alwaysOn>true</alwaysOn>
                <updateRate>0.0</updateRate>
                <cameraName>rrbot/camera1</cameraName>
                <imageTopicName>image_raw</imageTopicName>
                <cameraInfoTopicName>camera_info</cameraInfoTopicName>
                <frameName>camera_link</frameName>
                <hackBaseline>0.07</hackBaseline>
                <distortionK1>0.0</distortionK1>
                <distortionK2>0.0</distortionK2>
                <distortionK3>0.0</distortionK3>
                <distortionT1>0.0</distortionT1>
                <distortionT2>0.0</distortionT2>
            </plugin>
```

深度相机的(这个是kinetic 相机的)： 

**注意相机坐标系与世界坐标系的转换关系**

```xml
<plugin name="camera_plugin" filename="libgazebo_ros_openni_kinect.so">
          <baseline>0.2</baseline>
          <alwaysOn>true</alwaysOn>
          <!-- Keep this zero, update_rate in the parent <sensor> tag
            will control the frame rate. -->
          <updateRate>0.0</updateRate>
          <cameraName>camera_ir</cameraName>
          <imageTopicName>/camera/color/image_raw</imageTopicName>
          <cameraInfoTopicName>/camera/color/camera_info</cameraInfoTopicName>
          <depthImageTopicName>/camera/depth/image_raw</depthImageTopicName>
          <depthImageCameraInfoTopicName>/camera/depth/camera_info</depthImageCameraInfoTopicName>
          <pointCloudTopicName>/camera/depth/points</pointCloudTopicName>
          <frameName>camera_link</frameName>
          <pointCloudCutoff>0.5</pointCloudCutoff>
          <pointCloudCutoffMax>3.0</pointCloudCutoffMax>
          <distortionK1>0</distortionK1>
          <distortionK2>0</distortionK2>
          <distortionK3>0</distortionK3>
          <distortionT1>0</distortionT1>
          <distortionT2>0</distortionT2>
          <CxPrime>0</CxPrime>
          <Cx>0</Cx>
          <Cy>0</Cy>
          <focalLength>0</focalLength>
          <hackBaseline>0</hackBaseline>
   </plugin>
```

有关雷达注意选择是 GPU 还是 CPU

controllers

```bash
sudo apt-get install ros-melodic-effort-controllers
```

yocs nodelet

```bash
sudo apt-get install ros-melodic-yocs*
```
