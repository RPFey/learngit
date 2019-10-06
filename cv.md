# Low level

## 放大

### 最邻近插值法

选择最接近插值点的像素值作为该点的像素值。出现锯齿现象

### 双线性插值

A      B

​      P

C       D

现在AB 中根据加权平均得到 e 点的值， 同理得到 f 点值， 再由 e,f 得到 P 点值。(一律采用直线拟合)

### 双立方插值

将影响像素点扩大到16 个，采用非线性计算拟合

### 自适应插值

## 实现

将 目标图像 上的点换算到原图像上，然后采用取整运算找到周围的点，获取像素值计算。这样就不需要为整数倍。

## 缩小

### 等间隔采样

（其实可以与图像金字塔比较一下） 也就是等距离采样

### 均值采样

在 2*2 的方格中求取均值后作为采样值

实现方法都是先换算坐标，再求值

# middle

## zhang suen thinng

Zhang-Suen细化算法通常是一个迭代算法，整个迭代过程分为两步：

Step One：循环所有前景像素点，对符合如下条件的像素点标记为删除：

1.      2 <= N(p1) <=6

2.      S(P1) = 1

3.      P2 * P4 * P6 = 0

4.      P4 * P6 * P8 = 0

其中N(p1)表示跟P1相邻的8个像素点中，为前景像素点的个数

S(P1)表示从P2 ～ P9 ～ P2像素中出现0～1的累计次数，其中0表示背景，1表示前景。

0~1 代表从 0 到 1 的次数。

完整的P1 ～P9的像素位置与举例如下：

P9    P2    P3

P8    P1     P4

P7     P6    P5


其中 N(p1) = 4, S(P1) = 3, P2*P4*P6=0*0*0=0, P4*P6*P8=0*0*1=0, 不符合条件，无需标记为删除。

Step Two：跟Step One很类似，条件1、2完全一致，只是条件3、4稍微不同，满足如下条件的像素P1则标记为删除，条件如下：

1.      2 <= N(p1) <=6

2.      S(P1) = 1

3.      P2 * P4 * P8 = 0

4.      P2 * P6 * P8 = 0

循环上述两步骤，直到两步中都没有像素被标记为删除为止，输出的结果即为二值图像细化后的骨架。

关键在于采用 编码的方式 卷积核选择为 

1   2    4

8   0    16 

32  64  128    这样每一种情况对应一个单独的数，采用计算机枚举的方法判断所有情况。

## calcbackproject :

输出与目标图片一样大小的数组。目标图像中的像素点根据对模板颜色直方图的概率模型得到。推荐使用 hue saturation .

如果采用边缘信息做效果不好。尝试将 sobel 之后的图片做成 hist 再 calcbackproject，几乎不能匹配。

```python
# calculating object histogram
roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

# normalize histogram and apply backprojection
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

# Now convolute with circular disc
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(dst,-1,disc,dst)

# threshold and binary AND
ret,thresh = cv.threshold(dst,50,255,0)
thresh = cv.merge((thresh,thresh,thresh))
res = cv.bitwise_and(target,thresh)
```

opencv 实现方法 ： 在 test image 中的一个 pixel 得到 color information, 寻找其在 test image histgram 中所在的 bin (h, s). 再在 model image 中找到对应的 bin (h, s) 将其值写入 output 中对应点 (x, y) 中。这样相当于基于统计，得到该像素在 model image 中的概率。

## haar cascade:

### opencv_tools

安装opencv 时带有一些可以用来训练 haar cascade 的工具：

opencv_createsamples : 

-vec  <vec_file_name>   产生*.vec 文件 保存了对positive images 的信息

-bg  <background_file_name>  预先制作的 bg.txt 文件，是对 negative images 的信息

-info <collection_file_name>  获取 预先标注的数据 文件

-w  .... -h .... sample patch 的 长宽比

生成机制：

The object instances are taken from the given images, by cutting out the supplied bounding boxes from the original images. Then they are resized to target samples size (defined by `-w` and `-h`) and stored in output vec-file, defined by the `-vec` parameter. No distortion is applied, so the only affecting arguments are `-w`, `-h`, `-show` and `-num`.

（此命令也可以用来将一张图叠加到negative images生成大量的positive images ， 采用下面参数:)

-maxidev  ...   -maxxangle  ... 

利用 demo 中的 detectmulti 和 读取 xml 接口 来完成。

## SVM

save 模型时 ， 有 process finished with exit code 139 

这是因为写地址溢出了，只需要把 .xml 删掉再写就可以了

关于装载模型的问题， 

```python
svm = cv2.ml.SVM_load(cv2.samples.findFile('./svm.xml'))
```

这里采取 与 haar cascade 相同的装载方法，且要求版本为 4.1 以上。







