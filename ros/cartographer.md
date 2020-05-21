# Cartographer

官方有调参文档，针对使用。

node_main node_ope

map_builder_bridge

map_builder_interface / trajectory_builder_interface

```c++
// in node_main.cpp

int main(){
// ...
cartographer::Run();
}

```

NodeOption/TrajectoryOptions 加载 Lua 文件
Node 大类

SensorBridge：

HandeRangeFinder 实际加载雷达数据。
