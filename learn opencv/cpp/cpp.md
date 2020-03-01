写一些　cpp 中的用法

# vscode cmake

通过在 CmakeLists.txt 设置 set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

得到 build 下的 compile_commands.json , 然后在 c_cpp_properties.json 中加入 "compileCommands": "${workspaceFolder}/build/compile_commands.json" 便可以由 cmake 配置 c_cpp 索引

# vector 

## copy 
```c++
vector<int> list;
// 1. initialization
vector<int> copy_(list);
// 2. assgin (copy)
copy_.assign(list.begin(),list.end());
// 3. swap, the original one is empty
copy_.swap(list);
// 4. insert, insert the original one into the new vector
copy_.insert(copy_.end(), list.begin(), list.end());
// the first argument can change to decide where to insert
```

# map

一个键值对，相当于 python 中的字典 (自带顺序)

map<type1, type2> obj;  键与值的类型
```c++
// 构建

map<string, any> params ; // flann 中构建参数的方法
params["algorithm"] = algorithm ; 

// 使用
// 带有一个数据结构 pair 
```

# Boost

## boost::array

```c++
// 与　ros 消息转化　
boost::array<double,9> intrinc {msg->K}; // K 是float64[9]
// 与　mat 转化
cv::Mat temp_intri(3,3,CV_64FC1, &intrinc[0]);
```
