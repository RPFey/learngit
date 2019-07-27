# 环境搭建

从windows 上看，需要头文件的路径的导入（环境变量的设置）和连接库的配置（g++ 中的 dpkg 的配置）

g++ 编译：

`pkg-config opencv --cflags --libs`

# OPENCV 中的其他类

## Point

### point2d

这个类实例化后多用于图片中点的选择。

注意： point(j,i) 是指第j列，第i行。

# 色彩空间

cvtColor 转换色彩空间，可以在其他空间中对图片的亮度等等进行调整，再变换回RGB空间。三参数，指令参数 好像多为 COLOR_  开头。

在imread 中可以在读入中采用二参数转换为灰度图

# cv::mat

<https://blog.csdn.net/u010248552/article/details/79962132>

<https://www.bilibili.com/video/av29600072/?p=4>

## 初始化

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
Mat src(2,2,CV_8UC3,Scalar(0,0,255)); // 8UC3 就规定了是3通道的
// 创建图像并复值 ， Scalar 类用于赋值。
// 有些里面会有 Size 类， 也是直接实例化即可
//定义小数组
Mat kernel = (Mat_<float>(3,3)<<0,-1,0,-1,5,-1,0,-1,0);
```

## Mat结构

我们先来看看Mat结构，在你初始化一个Mat结构时，它的全部结构是这样的。
Mat类可以分为两个部分：矩阵头和指向像素数据的矩阵指针。
MAT

### flags

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

更细致的分析推荐这篇文章flags

作者：callback 
来源：CSDN 
原文：https://blog.csdn.net/u010248552/article/details/79962132 

### dims

为矩阵的维数 ， 一般大于等于2

### cols & rows & channels

作为mat 的属性 ， 返回列数,行数,通道数

### data

作为属性 ， 返回指向元素的指针

## 操作

### .at()

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

## .convertTo()

```c++
src.convertTo(dst,CV_32FC1); // 可以改变Mat 的类型
```



## 图像指针

```c++
image.ptr<uchar>(int i = 0)
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



## 采用模板子类 ：Mat_

在实例化Mat 的时候 , 指定类型：

```c++
Mat_<Vec3b> image;
// 调用时
image(j,i) = 100;
```

此时便不需要再用 .at 

# 图像变换

## 降采样

（信号处理）

## 高斯模糊

```c++
GaussianBlur(src,dst,Size(i,i),sigma);
```

### 2019.7.25

也算是看到了其一个新用法吧，检测交点。可以采用高斯模糊（假设直线为255，背景为0）。当相交的直线越多，那么锚点处的值越大。选择合适的区间可以找到不同数量的相交直线的交点。

有一定效果，但是一般。

注意： sigma 可以适当加大。

## 边缘检测

采用 canny 函数

## 轮廓检测

```c++
findContour(InputOutputArray image, OutputArrayofArrays contours, OutputArray hierarchy,
			int mode, int method, Point offset = Point())
```

输入的图像一般是经过canny 边缘检测的二值图像。

contours 是 

```c++
vector<vector<Point>>
```

每一组Point 点集就是一条轮廓。

(但是效果不见得很好，今天测试有700余条轮廓，但第一个轮廓只有一个点。 后来发现是线宽的问题。 2019.7.26)

hierarchy

```c++
vector<Vec4i> hierarchy
```

hierarchy 中的元素与contours 中的轮廓元素一一对应，其中的四个整数元素为contours中的索引。代表该轮廓的后一个轮廓，前一个轮廓，父轮廓，内轮廓。若没有则为-1。

opencv 中轮廓的关系：如果A 轮廓在B 的内部，则称A 为B 的子轮廓，B 为A 的父轮廓。

若两者并列，则为同级，可以有前后顺序。

最后一个是偏移量。

# 滤波与卷积

## 概念

锚点 : 核与原图像的对齐关系

## padding

采用 copyMakeBorder :

给定一张图片和上下左右的补丁像素值.

而 "逆变换" -- borderInterpolate

则是计算边界的像素值是由原图中哪个像素点给出

## filtering 

threshold  比较简单的滤波函数，能得到滤波后的矩阵

当不确定时，可以采用Otsu算法，自动决定最优阈值。

另一个是 adaptiveMethod 自适应阈值（应用于明暗差异变化较大），亮出的暗值大于暗处的亮值。

感觉是加了一层模糊

## conv

```c++
// 直接filter 2D
filter2D(src,output,-1,kernel); // 这样就可以保持一致了。 kernel 可以使用之前定义的小数组（2，2）
                                // 有自动填充三通道的功效。
```

对于多通道的图像，可以用 split 将通道拆分成单通道，单独卷积，再合并。

# 问题

1.无法读取视频

<https://blog.csdn.net/lingzhaoli/article/details/78711445>

ffmpeg 文件出现问题

闲话 : 文件名大小写也很重要,包括扩展名.

