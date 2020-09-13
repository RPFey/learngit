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