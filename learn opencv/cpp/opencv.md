# cv::mat

<https://blog.csdn.net/u010248552/article/details/79962132>

## 初始化

```c++
// 1、imread
Mat src = imread("csdn.png");

//2、create
Mat src;
if(src.empty())
{
	src.create(Size size, VC_8UC3);
}
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

}
else if(image.type()==8UC1){
image.at<cv::uchar>(j,i) = 100; // grey map use unchar(in cv namespace)
}
```

notation : in cv , the channel is B G R

### 采用模板子类 ：Mat_

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

## 边缘检测

采用 canny 函数