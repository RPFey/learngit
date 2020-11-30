# 环境搭建

从windows 上看，需要头文件的路径的导入（环境变量的设置）和连接库的配置（g++ 中的 dpkg 的配置）

g++ 编译：

`pkg-config opencv --cflags --libs`

opencv 包含一些老式的函数，对于参数类型有要求，尽量不要使用

cmake 中 变量赋值时中间不要有空格

```bash
-D OPENCV_EXTRA_MODULES_PATH=/opencv/...  ..
```

opencv_contrib 搭建 ：

报错 ： 找不到文件与目录

在 /usr/local/include 下面修改 因为opencv4 的 hpp 文件在 opencv4/opencv2/ 下 只 

```c++
#include "opencv2/opencv.hpp"
```

当然找不到， 把里面的 opencv2 复制到 /usr/local/include 下

报错 ： 未定义的类

这是连接库的问题， 先采用 pkg-config opencv --cflags --libs 查看连接库

```bash
apt-file search opencv.pc
```

找到相应的 pkg-config 文件 ， 修改其中 连接库的路径 与连接库 

报错 ： OPENCV_ENABLE_NONFREE 

没有 构建 nonfree 的函数， 要在 cmake 中重新 configure

```bash
cmake -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_BUILD_TYPE=Release -D OPENCV_ENABLE_NONFREE:BOOL=ON -D OPENCV_EXTRA_MODULES_PATH=/opencv/opencv_contrib-master/modules ..
```

注意 OPENCV_EXTRA_MODULES_PATH 为 opencv_contrib 下 modules 的路径

# 色彩空间

cvtColor 转换色彩空间，可以在其他空间中对图片的亮度等等进行调整，再变换回RGB空间。三参数，指令参数 好像多为 COLOR_  开头。

在imread 中可以在读入中采用二参数转换为灰度图

# 视频处理

视频处理采用 VideoCapture 对象。在 VideoCapture 类中以下的辅助函数。

采用 .isOpened() 方法检测是否正常打开。构造函数中为打开视频的名字或着整数，代表电脑上的摄像机。或是用 .open(....) 方法打开文件。

# 图像变换

## gamma 矫正

## 图像混合

```c++
addWeighted(inputArray src1,
            double alpha,
            inputArray src2,
            double beta,
            OutputArray dst,
            int dtype = -1);
```

可以用这个函数做矩阵的加法。（不过直接相加也可以，为什么用这个函数？对数乘也重定义了的。）

而且超过255就是255，没有溢出。

## harris corner detctor

## 边缘检测

sobel 算子： 离散的微分算子，计算梯度图像。（高斯平滑与微分求导，原因在于求导与高斯模糊可以结合后再与图像计算，都是卷积）

```c++
Mat kernel = (Mat_<CV_32FC1><<1,2,1,0,0,0,-1,-2,-1) // y
```

sobel()  api 中的 depth 是输出图像的类型： eg. CV_32FC1

一般采用 CV_16S / CV_32F  , 如果精度不够，对提取的特征会有影响。

```c++
convertScaleAbs(src,dst,alpha,beta); // 会改变图像的类型，dst 类型变为 CV_8UC3
saturate_cast<...>(...);
// 对卷积后的图像数据进行处理，能可视化
```

还有  ： 对比效果更强的scharr 算子

laplace 算子：计算二阶导数。 api: Laplacian(....)

采用 canny 函数

canny 之前 会自带 Gaussian Blur

## 轮廓检测

### findcontour()

```c++
findContour(InputOutputArray image, OutputArrayofArrays contours, OutputArray hierarchy,int mode, int method, Point offset = Point())
```

输入的图像一般是经过 canny 边缘检测的二值图像。( adaptiveThreshold 阈值化 )

contours 是 `vector<vector<Point>>`

每一组Point 点集就是一条轮廓。

(但是效果不见得很好，今天测试有700余条轮廓，但第一个轮廓只有一个点。 后来发现是线宽的问题。也可以用反锯齿消除。 2019.7.26)

hierarchy

```c++
vector<vector<Point>> contours
vector<Vec4i> hierarchy
```

hierarchy 中的元素与contours 中的轮廓元素一一对应，其中的四个整数元素为contours中的索引。代表该轮廓的后一个轮廓，前一个轮廓，父轮廓，内轮廓。若没有则为-1。

opencv 中轮廓的关系：如果A 轮廓在B 的内部，则称A 为B 的子轮廓，B 为A 的父轮廓。

若两者并列，则为同级，可以有前后顺序。

最后一个是偏移量。

### approxpolydp

这里是用多边形去拟合点的数据。

```c++
void approxPolyDP(InputArray curve,
    OutputArray approxCurve,
    double epsilon, // 是拟合的精度
    bool closed);
// 调用：
vector<vector<Point>> contours;
vector<vector<Point>> contour_poly(contours.size()); // 两者大小保持一致
approxPolyDP(Mat(contours[i]),contour_poly[i], 30, 1);
```

结果 contour_poly 中存放的是拟合的多边形的顶点坐标，

epsilon 越小，拟合的多边形越相近。

### minEnclosingcirlce

寻找最小的包络圆。

2019.8.1 当圆只有一部分时，此方法效果不好。

### contourArea

求凸边形的面积

## 模糊

模糊与平滑可以去除图像的噪声。（亮斑的去除）

### 均值模糊

```c++
blur(src,dst,Size(i,j),Point(-1,-1);
```

i,j 最好选择奇数。Point是锚点的选择。（-1，-1）是默认的中心点。

### 高斯模糊

```c++
GaussianBlur(src,dst,Size(i,i),sigmaX, sigmaY);
```

Size 要求为奇数是要求确定中间的锚点。保留信息较多。

2019.7.25

也算是看到了其一个新用法吧，检测交点。可以采用高斯模糊（假设直线为255，背景为0）。当相交的直线越多，那么锚点处的值越大。选择合适的区间可以找到不同数量的相交直线的交点。

有一定效果，但是一般。

注意： sigma 可以适当加大。

### 中值滤波

去除椒盐噪声。指图像中随机存在极大或极小的像素点。

```c++
medianBlur(Mat src,Mat dest, Size ksize)
```

### 双边滤波

边缘保留的滤波方法，将像素值差值大于某一阈值的边缘保留，不模糊。保留了图像的边缘信息。

卷积核看成由空域核与值域核的叠加。其中的空域核代表空间距离，随到中心点的距离增加而减小。值域核代表通道值的差，随着相差增大而减小（保留边缘）。两者的乘积为滤波的核。

```c++
bilateralFilter(src,dest,radius,sigma_color,sigma_space);
```

## 滤波与卷积

### 概念

锚点 : 核与原图像的对齐关系

### padding

在《计算机视觉》中处理范围问题时，认为超出的部分值全部为0。

采用 copyMakeBorder :

给定一张图片和上下左右的补丁像素值.

BORDER_WRAP 用另一侧的边填充，在于映图像的空间周期。

在封装的模糊函数和卷积函数（filter2D）中的bordertype 就是指此处的 type。

而 "逆变换" -- borderInterpolate

则是计算边界的像素值是由原图中哪个像素点给出

### filtering 

threshold  比较简单的滤波函数，能得到滤波后的矩阵

当不确定时，可以采用Otsu算法，自动决定最优阈值。

另一个是 adaptiveMethod 自适应阈值（应用于明暗差异变化较大），亮出的暗值大于暗处的亮值。

感觉是加了一层模糊

### conv

```c++
// 直接filter 2D
filter2D(src,output,ddepth,kernel,Point=(-1,-1),delta = 0,bordertype); // 这样就可以保持一致了。 kernel 可以使用之前定义的小数组（2，2）
                                // 有自动填充三通道的功效。
```

对于多通道的图像，可以用 split 将通道拆分成单通道，单独卷积，再合并。

kernel 核的元素必须和为1。否则overflow。

ddepth 可以改变输出的数据精度 eg. CV_32FC1 ; bordertype 则是边缘外推的方式。

## 形态学操作

腐蚀：拿一个结构元素去滑动窗口，将窗口覆盖下图像的最小值当作结构元素中心点的像素值；

膨胀：拿一个结构元素去滑动窗口，将窗口覆盖下图像的最大值当作结构元素中心点的像素值；

对小区域的操作。

开操作：先腐蚀后膨胀，可以去掉前景中小的对象（在腐蚀中消失，膨胀中无法复原）。

闭操作：先膨胀后腐蚀，可以填充前景中小的洞（洞在膨胀中补齐，腐蚀不会使其重现）。

形态学梯度:  用于刻画目标边界或边缘位于图像灰度级剧烈变化的区域，形态学梯度根据膨胀或者腐蚀与原图作差组合来实现增强结构元素领域中像素的强度，突出高亮区域的外围。

梯度分为四种：

基本梯度

————基本梯度是用膨胀后的图像减去腐蚀后的图像得到差值图像，称为梯度图像也是OpenCV中支持的计算形态学梯度的方法，而此方法得到梯度有被称为基本梯度。

内部梯度

————是用原图像减去腐蚀之后的图像得到差值图像，称为图像的内部梯度。

外部梯度

————是用图像膨胀之后再减去原来的图像得到的差值图像，称为图像的外部梯度。

方向梯度

————方向梯度是使用X方向与Y方向的直线作为结构元素之后得到图像梯度，用X方向直线做结构元素分别膨胀与腐蚀之后得到图像求差值之后称为X方向梯度，用Y方向直线做结构元素分别膨胀与腐蚀之后得到图像求差值之后称为Y方向梯度。

对于二值图像，可以用形态学梯度保留物体的边缘轮廓信息。也就是去掉填充得到轮廓。

黑帽与白帽：提取开闭操作中的反差值。黑帽是闭操作与源图像的差值，白帽则是开操作。

顶帽与黑帽:图像与 开操作/闭操作 的差。得到孤立的小片区域。

### 操作核的形状的选择：

选择线（1，x）与（x，1）可以将斜线全部去除。而矩形则可以将干扰线去掉（经过一次开操作就可以）。

也就是选择特定的形状可以单独得到此种形状。

###  工具

```c++
Mat kernel = getStructuringElement(MORPH_RECT,Size(m,n),Point(-1,-1)); // 获得操作的核
dilate(...)； //膨胀
erode(src,dst,kernel); // 腐蚀
morphologyEx(src,dst,operation,kernel); // 各种操作由 operation 决定
```

### 使用技巧

黑帽用于提取较暗的区域，而白帽用于提取较亮的区域。

## 二值化图像&图像阈值

只能是单通道八位图像。

<https://blog.csdn.net/u012566751/article/details/77046445>

```c++
threshold(src, dst, thresh, maxvalue, type)
/* type 在下面详细解释
   maxvalue 与 THRESH_BINARY 有关
*/

adaptiveThreshold(src , dst , maxValue , adaptivemethod , thresholdType , blocksize, C)
```

阈值的取法有很多,不一一讲述.只讨论 THRESH_BINARY 时, 大于阈值,则为maxvalue 否则为0.

THRESH_OTSU  自动寻找合适的阈值.   eg.（THRESH_OTSU|THRESH_BINARY）

adaptivemethod 没有指明阈值是多少,且要求必须为THRESH_BINARY / THRESH_BINARY_INV 是根据blocksize 这个正方形块中的像素值,采用均值或高斯均值的方法确定阈值.

在《学习opencv3 》中演示了adaptiveThreshold 函数，在图像有阴影时，后者比前者能提取更完整的图像。

```plain
normalize()
```

## 图像金字塔

具体见计算机视觉讨论理论

高斯不同(DOG):

将一张图片在不同参数下高斯模糊得到的结果相减,结果为高斯不同.

将不同层检测出的特征与圆图像匹配

### 工具

```c++
pyrUP(src,dst,Size(..,..)) / pryDown(...)
```

## 霍夫变换

前提是 canny 边缘检测完成。(获得二值图像)

霍夫变换是在 r - theta 空间中表示 经过 x-y 空间中一点处的所有直线。即经过一点处所有的直线在 r-theta 图上表现为一条曲线。后者中曲线的交点代表 x-y 空间中该 （r-theta）直线经过多个点。

```c++
HoughLines(src, lines, rho, theta, threshold, ...);
// 得到的是(theta,r) 表示的极坐标空间
HoughLinesP(src, lines, rho, theta, threshold, minLineLength = 0, maxLineGap = 0);
// 最终输出直线的两个点
// minLineLength 代表直线最小的长度(即直线至少包含的像素点数)
// maxLineGap 代表最大直线间隔，同一行中点与点之间的最大间隔
// rho, theta 代表在hough 空间中搜索的精度 推荐为 1.0(pixel) , CV_PI/180 (rad)
```

得到的 Vec4f 构建 两个 Point 对象， 直接用 line 函数绘图

由于离散化，使得斜线上的点可能不连续， 通过 maxLineGap 检测斜线，使其为一条。

hough circle transform 对噪声比较敏感， 先用中值滤波。

对于圆，由于 x,y,theta 可以表示一个圆，变换后的圆空间就是3维。（可以用于弧线的检测）

## 几何变换

对于图像在平面区域，有两种变换：

1. 仿射变换：基于 2*3 矩阵变换
2. 透视变换，单应性映射：基于 3*3 矩阵变换

仿射变换，给定 2*3 的变换矩阵。变换矩阵由三对对应点对构造。

```c++
// 稠密变换的函数是
void warpAffine( 	
    InputArray  src,
    OutputArray  dst,
    InputArray  M,
    Size  dsize,
    int  flags = INTER_LINEAR,
    int  borderMode = BORDER_CONSTANT,
    const Scalar &  borderValue = Scalar() 
)

// 获取变换矩阵
cv::Mat 
```

透视变换则是由四对对应点对构成


# 相机标定

## 实际应用

回答一个问题：

为什么opencv 标定中将实际点的坐标　Z 全部设为　０, 而棋盘中各点的相邻长度都设为1。

标定时得到的是实际的内参，这与外界坐标系的选取无关。由于内参矩阵(3,3)这个元素为１，保证了不同尺度下的唯一，改变的只是变换矩阵的大小。而当要知道实际物体的大小时，需要知道深度，换算为在相机坐标系中的坐标。

## 棋盘标定

主要在　board_size , 不计算最边上的一个角点。当没有找到与所给board_size 相同数量的角点时，findChessboardCorners 会返回０。

```c++
/* --- 相机中坐标获取 --- */
Size image_size;  　　　　　　　　　　　/* 图像的尺寸 */
Size board_size = Size(4,6);        /* 标定板上每行、列的角点数 */
vector<Point2f> image_points_buf;   /* 缓存每幅图像上检测到的角点 */
vector<vector<Point2f>> image_points_seq; /* 保存检测到的所有角点 */
// 找到的点都是
findChessboardCorners(imageInput,board_size,image_points_buf);
// 在每一个角点周围　(11,11)　区域内搜寻更精确的角点。传入为灰度图像
find4QuadCornerSubpix(view_gray,image_points_buf,Size(11,11)); 

/* --- 实际坐标获取，并得到内参矩阵　--- */
// 向　calibcamera 中传入的数据格式
vector<vector<Point3f>> object_points; 
vector<vector<Point2f>> image_points_seq;
Mat cameraMatrix=Mat(3,3,CV_32FC1,Scalar::all(0)); /* 摄像机内参数矩阵 */
vector<int> point_counts;  // 每幅图像中角点的数量
Mat distCoeffs=Mat(1,5,CV_32FC1,Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
vector<Mat> tvecsMat;  /* 每幅图像的旋转向量 */
vector<Mat> rvecsMat; /* 每幅图像的平移向量 */
calibrateCamera(object_points,image_points_seq,image_size,cameraMatrix,distCoeffs，rvecsMat,tvecsMat,0);
```

# C++ 讨论

## sort()

sort() 函数传入比较函数这个参数时

# 问题

1.无法读取视频

<https://blog.csdn.net/lingzhaoli/article/details/78711445>

ffmpeg 文件出现问题

闲话 : 文件名大小写也很重要,包括扩展名.

