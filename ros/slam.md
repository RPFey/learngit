# slam

## AMCL

AMCL 定位 2D 概率定位系统 采用激光雷达等定位

### 参数设置

laser_z_* 传感器模型中的激光参数设置

odom_alpha_* 里程计模型中参数

odom_model_type 设置里程计的模型

### 代码解读

AMCL Node 结构：

1. 读取参数
2. 获取tf 坐标变换(tf2_ros::TransformBroadcaster, tf2_ros::Buffer, tf2_ros::TransformListener)
3. 设置雷达接受回调函数(tf2_ros::MessageFilter) (主要处理函数)
4. 地图处理
5. 动态参数设置　(dynamic_reconfigure::Server)

在　src/amcl/sensors/amcl_odom.cpp 里面有运动更新(AMCLOdom::UpdateAction). AMCLLaser::UpdateSensor 是激光的更新。

* 粒子集的数据结构

粒子采用　pf_sample_t 动态数组维护，粒子集使用 pf_sample_set_t 作为顶层粒子集的封装。

```plain
| ---- pf_t
        | --- int min_samples
        | --- int max_samples # min and max number of samples
        | --- int current_set # keep two sets (previous and current), use `current` to denote the current set
        | --- pf_sample_set_t sets[2]
                | ---- int sample_count # number of samples
                | ---- pf_sample_t* samples # individual particle
                | ---- pf_kdtree_t* kdtree # kdtree encoding of histogram
                | ---- int cluster_count, cluster_max_count
                | ---- pf_cluster_t* clusters # cluster of particles
                | ---- mean, cov
        | ---- double w_slow, w_fast # running averages, slow and fast of likelihood
        | ---- pf_init_model_fn_t random_pose_fn # function to draw random pose samples
```

* 模拟随机采样

when draw randomly from a zero-mean Gauss Distribution, Use the polar form of the [Box-Muller Transformation](http://www.taygeta.com/random/gaussian.html)

```c++
    double pf_ran_gaussian(double sigma){
        do
        {
            do{ r=drand48(); } while(r==0.0);
            x1 = 2.0 * r - 1.0;
            do{ r=drand48(); } while(r==0.0);
            x2 = 2.0 * r - 1.0;
            w = x1*x1 + x2*x2;
        }while(w > 1.0 || w==0.0);
        return(sigma*x2*sqrt(-2.0*log(w)/w));
    }
```

重点函数 laserReceived：

1103-1144 是在记录是否有 base_link 到 laser_scan_frame_id 的变换，如果没有则在 lasers_ 中记录。lasers_ , frame_to_laser_ 两个构成索引的表。则直接在 frame_to_laser_ 中得到相应雷达变换的索引

1147-1153 将 odom 到 base_frame 的变换，由轮式里程计提供（相当于机器人走过的路程）

1158-1177 计算 pose 的变化， pose 是目前的位姿。(这里应该有外部提供了里程计信息), 这里 update 变量是为了防止移动距离过小（或者不移动）导致重采样失败引入。（具体见书重采样部分）

1179-1195 初始化粒子滤波器(如果没有的话)

1197-1214 运动更新部分（具体可以看笔记，概率机器人）

之后是观测更新。

* 将激光的观测数据转换到 base_frame 下

```c++
tf->transform(min_q, min_q, base_frame_id_); //1240
tf->transform(inc_q, inc_q, base_frame_id_); //1241
```

> lasers_update (array, bool) 中存放的应该是要更新的激光数据

* Update Sensor Data

> 代码中实现了书中的四种传感器模型

1. 得到所有的粒子权重
2. 每个粒子权重归一化 （概率表示）
3. update running averages of likelihood of samples (Prob Rob p258) (??)

* 重采样



## mapping

采用 gmapping 构建导航图，在rviz 中得到导航地图， ROS-Academy 中slam 有 gmapping launch

rosrun map_server map_saver -f mymap 保存生成的地图

gmapping 订阅雷达数据和坐标（tf）并发布到 /map 话题上， OccupancyGrid.msg

当出现　Messagefilter dropped 100% of messages　时，　问题在于 tf 树之间有问题，　订阅的消息没有确定的坐标转换关系。

上面的数值代表存在障碍物的概率， 0 free; 1 obstacle

map_server 生成 static_map 不能修改

tf 要求： laser -> base_link -> odom

configure parameters:

maxUrange : max usable data of range from lidar

minimumScore :

## localization

AMCL 定位；  蒙特卡洛定位

先预先生成随机的位姿，通过机器人的移动，滤去不可能的位姿。

## path_planner

Naviagtion 导航，包括路径规划算法。

frame_id 绑定在 map 这个frame上 ， resolution 代表一个像素点在实际中的距离

frame 中 data 直接是把图片压成一维了， width*height

1. 重新定位机器人， 2D pose estimation

2. set 2D nav goal

## Navigation

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

## rtabmap

rgb-d slam package
