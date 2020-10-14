# OpenCV 数据结构

## cv::mat

<https://blog.csdn.net/u010248552/article/details/79962132>

<https://www.bilibili.com/video/av29600072/?p=4>

### 初始化

```c++
// 1、imread
Mat src = imread("csdn.png");

//2、create
Mat src;
if(src.empty())
{
    src.create(Size size, VC_8UC3);
    // or:
    src.create(init.size() , init.type());
    //创建一张已知图像大小，类型的空图像
}

// 3. 类初始化
Mat src(2, 2, CV_8UC3, Scalar(0,0,255)); // 8UC3 就规定了是3通道的
// 创建图像并复值 ， Scalar 类用于赋值。
// 有些里面会有 Size 类， 也是直接实例化即可
//定义小数组
Mat kernel = (Mat_<float>(3,3)<<0, -1, 0, -1, 5, -1, 0, -1, 0);

// 指针初始化
cv::Mat img = cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM::CPU));
// 直接将指针指向数据存储地址，避免拷贝。但是应注意共享内存带来的后果。
```

### Mat结构

我们先来看看Mat结构，在你初始化一个Mat结构时，它的全部结构是这样的。
Mat类可以分为两个部分：矩阵头和指向像素数据的矩阵指针。
MAT

#### flags

mat_flags
从定义可以看出flags是int类型，共占32位，结合上图可以看出各位所代表的意思。

从低位到高位：

0-2位代表depth即数据类型（如CV_8U），OpenCV的数据类型共7类，故只需3位即可全部表示。

3-11位代表通道数channels，因为OpenCV默认最大通道数为512，故只需要9位即可全部表示，可参照下面求通道数的部分。

0-11位共同代表type即通道数和数据类型（如CV_8UC3）

12-13位暂没发现用处，也许是留着后用，待发现了再补上。

14位代表Mat的内存是否连续，一般由creat创建的mat均是连续的，如果是连续，将加快对数据的访问。

15位代表该Mat是否为某一个Mat的submatrix，一般通过ROI以及row()、col()、rowRange()、colRange()等得到的mat均为submatrix。

16-31代表magic signature，暂理解为用来区分Mat的类型，如果Mat和SparseMat

更细致的分析推荐这篇[文章](https://blog.csdn.net/u010248552/article/details/79962132)

#### dims

为矩阵的维数，一般大于等于2

#### cols & rows & channels

作为mat 的属性 ， 返回列数,行数,通道数

#### data

作为属性，返回指向元素的指针

### 操作

#### .at()

```c++
if (image.type()==8UC3){

image.at<cv::Vec3b>(j,i)[0] = 100; // color map use Vec3b as template , j,i is the location and [0] is the channel
Vec3b pixel = image.at<Vec3b>(j,i);
// pixel 就相当于一个vector 再索引就好了。
}
else if(image.type()==8UC1){
image.at<cv::uchar>(j,i) = 100; // grey map use unchar(in cv namespace)
}
```

notation : in cv , the channel is B G R

后面的类型要与Mat 匹配。

当把某一通道改为0时，可以明显感觉到 color constancy.

#### .convertTo()

```c++
src.convertTo(dst,CV_32FC1); // 可以改变Mat 的类型
```

### 图像指针

```c++
uint8_t* p = image.ptr<uint8_t>(0);
```

这里就是获得每一行头部的指针，内部代表类型，后面是行数，从0开始计数。

```c++
const uchar * current = image.ptr<uchar>(i) // 指向行首的指针
now  = current[j] // 获得(i,j)的值
// 在采用卷积操作时使用，在多行上同时处理。
saturate_cast</*type*/>(/*expression*/) //相当于一个滤波函数，type的范围内。
```

掩膜操作，可以提高图像的对比度。根据 《computer  vision》中的intensity 在空间中连续变化，那么较暗像素点的周围均为暗像素点，亮处亦然。

3*3 格子中，中间为5，四条边中点为-1，其余为0。

### 采用模板子类 ：Mat_

在实例化Mat 的时候 , 指定类型：

```c++
Mat_<Vec3b> image;
// 调用时
image(j,i) = 100;
```

此时便不需要再用 .at

## InputArray

这是一个接口类，可以是 `Mat`, `Mat_<T>`, `vector<T>`, `vector<vector<T>>`等作为参数。空矩阵用 `cv::noArray()` 或者 `cv::Mat()` 均可。

`_InputArray::getMat()` 获取 `Mat` 变量

## Point

x,y 是对象的属性

同时带有点乘的方法。

### point2d

这个类实例化后多用于图片中点的选择。

注意： point(j,i) 是指第j列，第i行。因为横着是 x 坐标， 竖着是 y 坐标 。

## Scalar

相当于一个向量，不超过4维。

## RNG（随机数生成）

```c++
RNG rng(12345); // 随机数，初始化种子
int a = rng.uniform(0,x); // 参数为随机数的范围
```
