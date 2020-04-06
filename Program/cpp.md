写一些　cpp 中的用法

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

# Boost

## boost::array

```c++
// 与　ros 消息转化
boost::array<double,9> intrinc {msg->K}; // K 是float64[9]
// 与　mat 转化
cv::Mat temp_intri(3,3,CV_64FC1, &intrinc[0]);
```

## boost::shared_ptr

智能指针管理，如果多个shared_ptr 共同管理一个对象时，只有所有指针全部脱离(引用计数清零)，对象才会被释放。

```c++
boost::shared_ptr<int> ps(1);
// .unique
ps.unique(); // 判断是否仅有一个指针指向所指对象
// .use_count
boost::shared_ptr<int> ps1 = ps;
ps.use_count(); // return 2 (2 pointers to the same object)
// .reset()
boost:;shared_ptr<int> p2(3);
p2.reset(p1); // 释放当前对象，并指向 p1
```

## boost::bind



# C++

## 关键字

override:

添加在基类的虚函数后，如果之后没有实现这些虚函数则会报错。

## 指针

指针存放着地址。指针的类型是对地址的解释。

例如都存放 4 byte 数据，若用 char 类型指针，则是四个字符；若用 int 类型指针，则是一个整数。

对内存的数据灵活处理。

可以用来处理 serialize 之后的数据；

同时可以对不同的数据做处理，例如 浮点数做位运算

```c++
float i = -0.123;
int *p = (int *) &i;
*p <<= 1
```

