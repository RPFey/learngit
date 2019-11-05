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

## metapackage

虚包， linux 软件包管理，底层软件系统。组合软件包。

Cmakelist 中并不生成 可执行文件，但是在package.xml 中会有其他包的运行依赖，便于安装

# structure

master ,node 启动时向 master 申请， master  管理通信

## node

launch 会自动 启动 roscore



## topic 

异步通信

massage 是topic 的内容，相当于一个类，而发布的消息相当于一个对象。定义在 .msg 文件中

rostopic pub 发布消息时， 若遇到消息中的变量赋值

则 x: 0.0 冒号后面空一格， 感觉像是字典构建

而 1:2:3 则不需要空格 (yaml 格式)



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



## .launch file

in ros wiki roslaunch/XML



## tf& URDF（unified robot description format）

ros 中的坐标变换标准 ，树状 tree

机器人各个关节处有坐标系（frame） ,  每个之间有关系， 形成树状结构

tf tree 之间必须保持联通。broadcaster 向关系中发布消息，确定关系，

/tf 下有多个节点发送消息

### Transformstamped.msg

指定从 frame_id -> child_frame_id 的变换 

### tf/tfMesssage.msg & tf2_msgs/TFMessage.msg

为上一数据结构的数组 

c++ 直接 sendTransform 发 vector 与 单个都可以

lookupTransform ： 时间戳问题： 填入 ros::Time(0), 表示最近一帧的

### urdf 

.udrf  描述机器人

link 部件/ joint 关节（link 连接关系）

link :

inertial/ visual/ collision

joint : 父子节点，变换关系

## slam

AMCL 定位

odometry 定位，



Naviagtion 导航，包括路径规划算法。

nav_msgs/OccupancyGrid :

frame_id 绑定在 map frame上 ， resolution 代表一个像素点在实际中的距离

frame 中 data 直接是把图片压成一维了， width*height

### OccupancyGrid.msg

上面的数值代表存在障碍物的概率





### Navigation

move_base 中心节点， 中间的插件只需要指定算法即可

外界代表需要提供的信息： /tf   /odom  /map  /sensor

输出： cmd_vel 

move_base :

全局规划， 只考虑地图上静态的障碍物（已知）； 局部规划： 动态； recovery: 处理异常

需要 Base Local Planner/ Base global planner/ recovery behavior (指定， 继承了nav_core )



costmap  (插件)

两张： （global/local） ;  

static layer : 订阅map topic ; obstacle layer : 动态添加  ; inflation layer : 膨胀障碍物，确定机器人安全范围 

Mapserver 直接提供建好的地图。

my_map.yaml 表示地图的参数， *.pgm 保存地图 



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



