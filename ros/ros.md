# ENV

## to eclipse 

catkin_make --force-cmake -G"Eclipse CDT4 - Unix Makefiles" DCMAKE_VUILD_TYPE=Debug -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j8

## vscode

安装ros 插件,  在命令行中用 ros:update c++ propertities

编译时输入得到信息

```bash
catkin_make -DCMAKE_EXPORT_COMPILE_COMMANDS=Yes
```

c_cpp_properties:

```json
"compileCommands": "${workspaceFolder}/build/compile_commands.json"
```

会得到编译时的其它引用

tasks.json:

```json
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558 
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "catkin_make", //代表提示的描述性信息, 或者说这个task 的名字
            "type": "shell",  //可以选择shell或者process,如果是shell代码是在shell里面运行一个命令，如果是process代表作为一个进程来运行
            "command": "catkin_make -DCMAKE_BUILD_TYPE=Debug",//这个是我们需要运行的命令，在bash 中
            "args": [],//如果需要在命令后面加一些后缀，可以写在这里，比如-DCATKIN_WHITELIST_PACKAGES=“pac1;pac2”
            "group": {"kind":"build","isDefault":true},
            "presentation": {
                "reveal": "always"//可选always或者silence，代表是否输出信息
            },
            "problemMatcher": "$catkin-gcc"
        }
    ]
}
```

launch.json:

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [ 
        {
            "name": "(gdb) Launch",　　 // 配置名称，将会在调试配置下拉列表中显示
            "type": "cppdbg",　　　// 调试器类型 该值自动生成
            "request": "launch",　　 // 调试方式,还可以选择attach
            "program": "${workspaceFolder}/devel/lib/rosopencv/svm",　　//要调试的程序（完整路径，支持相对路径）
            "args": [],　// 传递给上面程序的参数，没有参数留空即可
            "stopAtEntry": false,　// 是否停在程序入口点（停在main函数开始）
            "cwd": "${workspaceFolder}",　// 调试程序时的工作目录
            "environment": [],//针对调试的程序，要添加到环境中的环境变量. 例如: [ { "name": "squid", "value": "clam" } ]
            "externalConsole": false, //如果设置为true，则为应用程序启动外部控制台。 如果为false，则不会启动控制台，并使用VS Code的内置调试控制台。
            "MIMode": "gdb",　 // VSCode要使用的调试工具
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "catkin_make", ///////// 这个重要，需要与task中的label相同, 执行task
        }
    ]
}
```



## python-interpreter

指定解释器后会与ros 原有的解释器的 site-packages 路径冲突，因此在 导入一些外部包之前，

```python
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
```

在执行.py 文件之前，需要

chmod +x *.py

touch *.py

在 Cmakelist.txt 中 ：

```
catkin_install_python(PROGRAMS
   py/hog-svm.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
 
 catkin_python_setup() 
 # 如果提供了python 模块 （包含 setup.py） 加上。
```

就可以直接rosrun pkg *.py 了

### 指定解释器

在虚拟环境下安装 

pip install catkin-tools rospkg rospy 

在执行 rosrun 之前激活这个环境

在 *.py  第一行：

#!/usr/bin/env python3

## 消息传输

使用 numpy 传输时，遵循下列方法

```python
# publish or client node
msg.data = data.tostring()

# service or subscirber node
data = res.data.fromstring()
data = np.reshape(data, (..,..))
```

自定义的 msg/srv 在生成的 dist-package 中会有类型，可以自己看看



消息引用：

```python
import sensor_msgs

img_msgs = sensor_msgs.msgs.Image()
# 一般是作为一个类，类中属性的名称与 msg 一致
```



## problem

编译时报错： /usr/bin/env 'python\r'

这是由于文件在 windows 系统中重新编码过， 在linux 中多了一个 \r 

最好的方法就是直接从 github 上 clone

或者在命令行输入 :%s/^M//%g



PYTHONPATH 中一定要有指向系统 python2.7 dist-packages 的路径，否则会因为导入的 yaml 包不同而产生问题。（/usr/local/bin/python2.7/dist-packages）



# ROS

## 主要命令 

rospack / rosnode / rosmsg / rosservice / rosmsg / rossrv / rosparam

catkin_make 之后 要 source ./devel/setup.bash

刷新环境变量，才能找到包（rospack_list ... ）

package 是 catkin 编译的基本单元，递归查找每一个 package, 每一个package 不一定要在同一目录中

package 包含多个可执行文件 （节点）

## package

package 下有 Cmakelist.txt 与 package.xml , 定义 package

Cmakelist.txt 确定编译规则

package.xml 相当于 包的描述 （主要修改 build_depend & run_depend）

manifest.xml 是 rosbuild 下的包描述。

srv, msg , action 在包中放在相应文件夹下，作为自定义。

*.luanch, *.yaml(配置文件)

```
rospack find [package_name]
rospack list 

roscd [package_name]

rosls [package_name] # 列出pkg 下的文件信息

rosed [package_name] [file_name] # 编辑包下文件

catkin_create_pkg <package_name> [deps]
deps std_msgs nav_msgs # 具体 msg 在 rosmsg list 中查看

rosdep install [package_name]  # 安装依赖  clone 下的pkg 需要安装， 由 package.xml 指导

rospack list | grep [...] # 可以过滤字符串
```



不同package 之间引用，需要其余包的message, 源文件之类的。

这都需要 find_package (... )

对于前者，include (${catkin_INCLUDE_DIR}) 可以引用生成的message 头文件， 

对于后者， 可以在前者包中生成库文件(.so)，然后引用即可 



2020.1.15

最近遇到一个与launch 有关的， roslaunch 时报错： 

invalid \<param>  tag : cannot load command parameter [rosversion] : returned with code [1]

这个是因为在更改 ROS_PACKAGE_PATH 时错误

## metapackage

虚包， linux 软件包管理，底层软件系统。组合软件包。

Cmakelist 中并不生成 可执行文件，但是在package.xml 中会有其他包的运行依赖，便于安装

# structure

master ,node 启动时向 master 申请， master  管理通信

## node

launch 会自动 启动 roscore

这里提一下多线程，有时发布数据较快而处理较慢，导致遗漏数据，可以考虑采用多线程，把数据缓存再处理。

单线程会将数据放入缓冲区，而如果之前缓冲区过长，导致数据长时间不能更新。

nh.param(name , value, default)  This method tries to retrieve the indicated parameter value from the parameter server, storing the result in param_val. If the value cannot be retrieved from the server, default_val is used instead.



## topic 

异步通信

massage 是topic 的内容，相当于一个类，而发布的消息相当于一个对象。定义在 .msg 文件中

rostopic pub 发布消息时， 若遇到消息中的变量赋值

则 x: 0.0 冒号后面空一格， 感觉像是字典构建

而 1:2:3 则不需要空格 (yaml 格式)

这里新加入使用socket 传输。ros 通信其实是用msg 类中的 serialize 方法将消息序列化，发送出去。接收端deserialize 恢复成原来格式。注意： 接收端的缓存空间！

ros 中　float32[] 可以用　vector 接受。而float32[9] 要用boost::array 接受。具体见　./learn opencv/cpp/cpp.md 中boost 库详解

## service

‘’相当于间断的发布消息‘’

request - reply 模型

client 发布消息后， 会在原地等待 service , 远程过程调用 （PPC）服务器端服务，调用另一个node的函数

service 定义 ：

... 

[request msg]

/ -----

....

[reply msg]

rossrv show [ rosservice info 下 type 后的类型]

rosservice call [service-name] "param: value"



## implementation

通过 gencpp,genpy 生成指定的文件，方便调用。

在 CmakeList.txt 中： 

```
add_message_files(...)
add_service_files(...)
..

generate_messages()

# 必须在 caikin_package 之前调用, catkin_package 的 CATKIN_DEPENDS 后加上 message_runtime
# 必须在 find_package 中找 message_generation, 且在 package.xml 中加上编译依赖与运行依赖。
```



## parameter server

存储参数字典 ， 存储配置

rosparam 查看

launch 文件中：

param name="..." value="..."

param name='....' command="...[执行文件] ...[参数文件]"

可执行文件得到参数文件作为参数后返回的值作为 param 的值

rosparam file="..."  command="load" 加载文件作为参数



还有一个namespace 的概念 ：

节点和话题都有自己的命名空间，一般情况下都在所在命名空间中进行通信， 还有对应的参数（和C++ 很像）

一般作为命令行参数传入 ：

\__name:= ...     ;  __ns:=.....

而 ros::NodeHandle nh('~') 代表私有的命名空间

## .launch file

in ros wiki roslaunch/XML

rosparam file = "..../ .. .yaml" command="load" 从其余配置文件导入参数

一般在

```xml
<node .. >
<rosparam file="..." coommand="...">
</node>
<include file="*.launch" /> launch 文件
```

remap ：　可以映射不同的话题，　将原本订阅/发布的话题改变成另一个

即可向节点中传入参数

## tf

ros 中的坐标变换标准 ，树状 tree, 使得不同sensor 得到的数据坐标能转换到同一坐标系下

机器人各个关节处有坐标系（frame） ,  每个之间有关系， 形成树状结构

tf tree 之间必须保持联通。broadcaster 向关系中发布消息，确定关系，

/tf 下有多个节点发送消息

eg. base_link to lidar 

Transformstamped.msg

指定从 frame_id -> child_frame_id 的变换 

tf/tfMesssage.msg & tf2_msgs/TFMessage.msg

为上一数据结构的数组 

c++ 直接 send Transform 发 vector 与 单个都可以

lookupTransform ： 时间戳问题： 填入 ros::Time(0), 表示最近一帧的



## slam

AMCL 定位 2D 概率定位系统 采用激光雷达等定位

### mapping

采用 gmapping 构建导航图，在rviz 中得到导航地图， ROS-Academy 中slam 有 gmapping launch 

rosrun map_server map_saver -f mymap 保存生成的地图 

gmapping 订阅雷达数据和坐标（tf）并发布到 /map 话题上， OccupancyGrid.msg

当出现　Messagefilter dropped 100% of messages　时，　问题在于 tf 树之间有问题，　订阅的消息没有确定的坐标转换关系。

上面的数值代表存在障碍物的概率， 0 free; 1 obstacle

map_server 生成 static_map 不能修改

tf 要求： laser -> base_link -> odom

configure parameters:

 maxUrange : max usable data of range from lidar

 minimumScore : ? 



### localization 

AMCL 定位；  蒙特卡洛定位

先预先生成随机的位姿，通过机器人的移动，滤去不可能的位姿。



### path planner

Naviagtion 导航，包括路径规划算法。

frame_id 绑定在 map 这个frame上 ， resolution 代表一个像素点在实际中的距离

frame 中 data 直接是把图片压成一维了， width*height

\1. 重新定位机器人， 2D pose estimation

\2. set 2D nav goal 

Navigation

move_base 中心节点， 中间的插件只需要指定算法即可。需要 Base Local Planner/ Base global planner/ recovery behavior (指定， 继承了nav_core )。当move_base 接受到goal后会连接其它组件，最后发送/cmd_vel 

move_base 实际上是一个 action_server, 接受goal pose, 所以用 rviz 设置2D nav goal 实际上是发布了一条消息。

service : /make_plan 只提供路径，而不移动

话题是 /move_base/goal, 通过发布来设定goal。

外界代表需要提供的信息： /tf   /odom  /map  /sensor

全局规划， 只考虑地图上静态的障碍物（已知）； 局部规划： 动态； recovery: 处理异常

parameter:  对nav_fn costmap planner 的参数

controller_frequency : 控制向base_controller 发送消息的频率。 

Tolerance parameters : 机器人的位姿与设定的位姿相差的允许值。

sim_time : base_local_planner 估计路径的长短

costmap

两张： （global/local） ;  global planner 采用static map 进行路径规划， 不会对sensor 的数据处理。有三层； 

static layer : 订阅map topic ; obstacle layer : 动态添加，避障  ; inflation layer : 膨胀障碍物，确定机器人安全范围 

local planner 在运动中会执行避障操作，并达到目的地。local planner 有不同选择 

base_local_planner : 随机选择一些允许的位移，并计算每条位移的结果。选择结果最好的。; bwa_local_planner , 

navfn(extension) , A* 迪杰斯特拉 / carrot planner , 可以根据障碍物设定

### rtabmap

rgb-d slam package

<<<<<<< HEAD:ros.md
### procedure

1. 检查数据接受是否正常

=======
### pointcloud_to_laserscan ＆　depthimage_to_laserscan
>>>>>>> c5421b567002ae14d780aabb10688a097e84d827:ros/ros.md

convert pointcloud data to laser scan data　

### ros&opencv

ros 中的cvImage

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

# ROSPY&ROSCPP

## rospy

publisher 初始化时， 设置queue_size 为较小整数， None 表示同步通信

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

<http://gazebosim.org/tutorials?tut=sensor_noise&cat=sensors>

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

其实 camera 是 model 下的一个属性

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

collision & visual 

visual : the visual part of the model 

collision: use simpler collision model to reduce computation time ("hit box")

基本思路：

collision 模块

visual 模块

## urdf  & sdf

具体文件在 robot_sim_demo 下的　urdf/ *.urdf.xacro 中，　可以看到各个　frame　之间的转换 

.udrf  描述机器人

link 部件/ joint 关节（link 连接关系）

link :

inertial/ visual/ collision

joint : 父子节点，变换关系



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







