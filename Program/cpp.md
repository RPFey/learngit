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

* override

添加在基类的虚函数后，如果之后没有实现这些虚函数则会报错。

* extern "C"

正确实现 c++ 调用其他 C 语言代码。指示编译器这部分代码按照 C 语言编译。这主要是因为编译 C 函数时不会带上函数的参数类型（C 语言不支持函数重载）

* attribute

\__attribute__((visibility('default'))):

控制共享库 (.so) 输出符号。'default' 代表其可以被导出。'hidden' 则不行。

* virtual

虚函数。在基类中声明时需要加 virtual 关键字，派生类中重新定义时加不加都可以。

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

## 函数指针

这里其实很像插件式设计，引用 AMCL 中的　AMCLLaser::UpdateSensor

```c++
// 申明函数指针
typedef double (*pf_sensor_model_fn_t) (void* sensor_data, struct pf_sample_set_t* set);

// definition
double AMCL::BeamModel(AMCLLaser* data, pf_sample_set_t* set){
    // ...
}

// function
void pf_update_sensor(pf_t *pf, pf_sensor_model_fn_t sensor_fn, void* sensor_data){
    // definition
    // if you want to call the function :
    (*sensor_fn)(pf, sensor_data);
}

// call
pf_update_sensor(pf, (pf_sensor_model_fn_t) BeamModel, data);
```
