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

Load functions create pointers and set it to sensors

```shell
gzserver -s <plugin_filename>
```

plugins 分为： world, model, sensor, system, visual, gui

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

Load 中 _sdf是 导入的 sdf 文件，含有标签信息

### model

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
      // simulation iteration.
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

在赋予物体运动特性时，注意把 static 这个标签设置为 false

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
## urdf  & sdf

[sdf_reference](http://gazebosim.org/sdf)

具体文件在 robot_sim_demo 下的　urdf/ *.urdf.xacro 中，　可以看到各个　frame　之间的转换 

udrf  描述机器人: 多用在ROS 下，需要将其修改才能在 gazbo 中使用。 

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

## With ROS

在与ROS 通信时，需要加入特定的插件，才能在 ROS 中收到相关信息。[link](http://gazebosim.org/tutorials?tut=ros_gzplugins&cat=connect_ros)

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
