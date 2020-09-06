# C++

## 编译相关

### 动态库

编译时，需要在 `.o` 文件后加上 `-fPIC` 选项。生成时，加上 `-shared -o libname.so`。

连接时，`-lname -L.` 连接。

`LD_LIBRARY_PATH` 动态链接库路径。程序运行时，会在该变量路径下寻找连接的动态链接库，需要更新。

> TensorRT 官方代码可以直接用动态链接库方式迁移到 ROS 工程下

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

### 函数指针

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

## 面向对象

c++ 派生类构造时，会调用基类的构造函数。

```c++
class Base ;

class Child : public Base ;


// 调用基类的构造函数并可以传入参数
Child::Child() : Base() {

}
```
