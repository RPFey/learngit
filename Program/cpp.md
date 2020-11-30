# C++

## 编译相关

### 动态库

编译时，需要在 `.o` 文件后加上 `-fPIC` 选项。生成时，加上 `-shared -o libname.so`。

连接时，`-lname -L.` 连接。之后使用 `ldd` 命令查看是否连接成功。

`LD_LIBRARY_PATH` 动态链接库路径。程序运行时，会在该变量路径下寻找连接的动态链接库，需要更新。

> TensorRT 官方代码可以直接用动态链接库方式迁移到 ROS 工程下

## 关键字

* override

添加在基类的虚函数后，如果之后没有实现这些虚函数则会报错。

* extern "C"

正确实现 c++ 调用其他 C 语言代码。指示编译器这部分代码按照 C 语言编译。这主要是因为编译 C 函数时不会带上函数的参数类型（C 语言不支持函数重载）

* attribute

```c++
__attribute__((visibility('default')));
```

控制共享库 (.so) 输出符号。'default' 代表其可以被导出。'hidden' 则不行。

* virtual

虚函数。在基类中声明时需要加 virtual 关键字，派生类中重新定义时加不加都可以。

* static & extern

static variable 和 static function 类似于一个源文件中的“私有”成员，不会被其他源文件中使用。也就是可以在两个不同源文件中定义的同名 static variable / function ，而不会报重定义的错误。
> 由于 include 是直接将头文件拷贝粘贴。所以头文件里定义的 `static` 变量会分别在不同引用的源文件中，互不干扰。但是头文件中定义的一般变量会分别在源文件中定义，编译器会报重定义的错。

`static` 在类中是被所有实例化的对象共享的，但是在使用前需要在类外再次声明。可以用 `CLASS::STATIC_VARIABLE` 或者 `CLASS::STATIC_FUNCTION` 来使用，更像是在 namespace 中的变量。**static function cannot use non-static member**
>these variables belong to a structure or class, But being static, they are not a part of the object of the class. So when an instance of a class is created, all the members of the class which are not static are declared, because static members are not a part of any instance, so to use them, we have to bring them to a scope from where they can be accessed
 
extern 是连接外部变量。`A.cpp` 想使用 `B.cpp` 中定义的同一变量则需要使用`extern`。函数的话可以使用头文件来完成。

* static_cast

显示类型转换。转换是在编译时，而不是运行时。编译时会检查转换是否成功。

```c++
float c = 1.0;
int b = static_cast<int>(a);
```

static_cast 比 c-type cast 更加严格。比如将 `char *` 转换为 `int *` 不被允许。
> 这是由于前者是 1 字节，后者是 4 字节，将会有 3 个字节是未定义的。

static_cast 防止派生类指针转换为**私有的**基类指针。向 `void*` 或由 `void*` 转换时，最好用 static_cast。

## enum

> `enum` gives names to a set of integers.

```c++
enum Example : unsigned char // member will be char
{
    A, B, C
};

Example value = A;
```

`enum` 有些类似于类，后面是类型名。`value` 只能赋 `Example` 中的值，但是能够与整形值比较。`enum` 中的名字可以赋值给整型变量（毕竟它们也是整数）。`enum` 里面会默认赋值。

## 函数

### 默认赋值

在 (.h) 文件中函数声明使用缺省值，在 (.cpp) 文件函数定义中不需要使用。

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

## 类

c++ 派生类构造时，会调用基类的构造函数。

```c++
class Base ;

class Child : public Base ;

// 调用基类的构造函数并可以传入参数
Child::Child() : Base() {

}
```

### Inheritance and Dynamic Memory Allocation

When the base class uses dynamic memory allocation,

* if the derived class doesn't use dynamic memry allocation
 
you don't need to explicitly define those constructor & destructor.

for destructor, it calls the base-class destructor after executing its own code.

for copy constructor, memberwise copying uses the form of copying that is defined for the data type in question. (that is defined uniquely in the derived class.) Copying a class member or an inherited class component is done using the copy construtor. **It is same for assignment**.

* if the derived class uses dynamic memry allocation

As the derived class destructor automatically calls the base-class destrcutor, it just needs to clean up the allocated memory of the derived class. Base-class destructor will clean up after what the derived-class constructor do.

```c++
class baseDMA {};
class hasDMA : public baseDMA{};

// destructor
baseDMA::~baseDMA{
    delete []label;
}
hasDMA::~hasDMA{
    delete []style;
}

// copy constructor
baseDMA::baseDMA(const baseDMA& rs){
    label = new char[std::strlen(rs.label) + 1];
    std::strcpy(label, rs.label);
    rating = rs.rating;
}

// base 的引用可以指向 derived
hasDMA::hasDMA(const hasDMA& hs)
    : baseDMA(hs)
{
    style = new char[std::strlen(hs.style) + 1];
    std::strcpy(style, hs.style);
}

// assignment
baseDMA & baseDMA::operator= (const baseDMA & rs){
    if (this == &rs)
        return *this;
    delete []label;
    label = new[std::strlen(rs.label) + 1];
    std::strcpy(label, rs.label);
    rating = rs.rating;
    return *this;
}  

hasDMA & hasDMA::operator= (const hasDMA & rs){
    if( this == &rs)
        return *this;
    baseDMA::operator=(hs);  // use the `operator=` explicit call
    delete [] style;
    style = new char[std::strlen(hs.style) + 1];
    std::strcpy(style, hs.style);
    return *this;
}
```

### type cast

如果希望对类进行类型转换，可以用 `operator` 实现。

```c++
class Int {
    int x;
    public:
        operator string () {
            return to_string(x);
        }
}

Int obj(20);
string a = static_cast<string>(obj);
```

### const member function

```c++
class Int {
    int x;
    public:
        void print () const {
            cout<<x;
        }
}
```

only const object can call const member function. const object cannot call normal member function. const member function cannot change the value of the class members.

### nested class

The class decalred with in another is called a nested class. eg.

```c++
class Queue
{
    // Node is a nested class definition local to this class
    class Node{
        int item;
        Node* next;
        Node(const int &i):item(i), next(0) {}
    }
}

// you can difine it outside the scope
Queue::Node::Node(const int &i):item(i), next(0) {}
```

* Scope

1. private: it is **only** known to that second class.
2. protected: it is known to the second class and classes derived from the second class
3. public: to all.

You should use a class qualifier outside the class scope when declaring the nested class (or structures/enums) like `BaseClass::NestedClass object`

### exception

Template:

```c++
try{
    // code here
    // throw ...
} catch ( ){
    // deal the things
    // call abort() if necessary
    // std::abort() is file <cstdlib>
}
```

if the objects thrown by the `throw` **matches** the type in the `catch` branket, the code inside the `catch` brace will be executed. eg.

```c++
try{
    // code here
    throw "error here";
} catch (char* err){
    // deal the things
    // call abort() if necessary
}
```

throw 抛出的也可以是对象。这样可以处理不同类别的异常。（与 python 中 Exception 类相同）

```c++
class A

try{
    // code here
    throw A();
} catch (A& err){
    // deal the error type A
    // call abort() if necessary
} catch (B& err){
    // deal the error type B
}
```

这里，`err` 尽管是引用，但也是拷贝，并且采用基类的引用，可以抛出不同的派生类。并且可以分级处理。

```c++
class bad_1 ;
class bad_2 : public bad_1 {};
class bad_3 : public bad_2 {};

void duper()
{
    if(no){
        throw bad_1();
    }
    if (rat){
        throw bad_2();
    }
    if (raw){
        throw bad_3();
    }
}

try {
    duper();
} catch(bad_3& err) {/*statements*/}
catch(bad_2& err) {/*statements*/}
catch(bad_1& err) {/*statements*/}
catch(...) // catch any exception
```

> 在函数后面添加 `noexcept` 关键字表明函数不会抛出异常。函数 `noexcept()` 判断操作数是否会抛出异常。

#### Exception Class

头文件`<exception>` 中有基类 `std::exception`，其虚函数 `const char* what()` 返回报错的字符串。

* `new` & `bad_alloc` error

you can use `new (nothrow) int[100]` to get the null pointer if you want.

```c++
#include <iostream>
#inlcude <new>
#include <cstdlib>

struct Big{
    double stuff[1000];
};

int main() {
    Big* pb;
    try {
        pb = new Big[1000];
    } catch (bad_alloc& ba){
        std::cout<<ba.what();
        exit(EXIT_FAILURE);
    }
    delete[] pb;
    return 0;
}
```

### I/O

* Binary File

c++ 中数据使用**小端存储**

```c++
ofstream f("data.bin", ios_base::out | ios_base::app | ios_base::binary);
f.write((char *)&data, sizeof(data));
```

`.write()` copies a specified number of bytes from memory to a file.
> `.read()` 使用方式相同。

* End of File

如果 `fin>>` 不能再读入数据了，才发现到了文件结尾，这时才给流设定文件结尾的标志，此后调用 `.eof()` 时，才返回真。也就是说，eof在读取完最后一个数据后，仍是 `False` ，当再次试图读一个数据时，由于发现没数据可读了，才知道到末尾了，此时才修改标志，`eof` 变为 `True`。

```c++
// 可以用如下方法持续读直到末尾。
if (fin.read());

while (fin.read()){

}
```

采用 `fin.gcount()` 获取读入字节数， `fin.clear()` 

* Input/Output File Mode

use `fstream` class. `.seekg()` moves the input pointer, `.seekp()` moves the output pointer.

```c++
fstream f("data.bin", ios_base::in | ios_base::out | ios_base::binary);
f.seekg(30, ios_base::beg); // 30 bytes beyond the beginning
f.seekg(-1, ios_base::cur); // 1 byte back up
f.seekg(0, ios_base::end); // the end
```
