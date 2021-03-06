# OPENCV_ADVANCE

## QRCODE detection

qrcode 检测的核心模块在 `QRDetect` 类中。

* init

先是缩放到 512, 并进行自适应滤波，完成对图像的预处理。

* localize

1. searchHorizontalLines

在水平方向上寻找比例为 1:1:3:1:1 的线段，记录行数。返回 `vector<Vec3d>` 按照 `起始位置`，`行数`，`总长度`格式。

2. separateVerticalLines

找到垂直方向上线段比例为 2:3:3:2:3:3 的点。返回 `vector<Point2f>` 格式 `x`, `y`

3. kmeans 聚类，

将其划分为 3 类 (qrcode 三个标定框)

至此，三个角的标定框全部检测完成。后面是固定标定框的顺序(fixationPoints)，缩放以及计算大小以滤除过小的方框。

从 `QRCodeDetector::detect` 中发现:

```c++
if (!qrdet.localization()) { return false; }
if (!qrdet.computeTransformationPoints()) { return false; }
vector<Point2f> pnts2f = qrdet.getTransformationPoints();
```

输出四个点经过调整，顺序为 `左上，右上，左下，右下`。

## [Aruco Markers](https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html)

Aruco is a kind of ￼binary square fiducial markers. Each corner is identified unequivocally for detection of the original pose.

* Marker creation

Use the predefined dictionary to generate markers with a given id.

* Marker detection

在函数 `_findMarkerContours()` 中检测预先的标定框，筛选过程如下:
> \* 代表需要关注修改

1. `adaptivethreshold` 二值化。
2. `findContours` 寻找凸多边形。
3. 设定 `minPerimeterPixels` 与 `maxPerimeterPixels`，寻找 contours 点数在这之间的。
4. `approxPolyDP` 近似凸多边形。寻找矩形 （*），且用 `isContourConvex` 寻找凸多边形。
5. `minCornerDistanceRate` 筛选各顶点之间的距离。(\*, 注意小物体设置)
6. `minDistanceToBorder` 忽略靠近图像边界的侯选框。

```python
inpuImage.copyTo(outputImage)
```

## FEATURE EXTRACTION AND MATCHING

使用 sift 特征

xfeature2d.sift_create()

## selective search

```python
# create Selective Search Segmentation Object using default parameters
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# set input image on which we will run segmentation
ss.setBaseImage(im)

# Switch to fast but low recall Selective Search method
ss.switchToSelectiveSearchFast()

# another mode is accurate : ss.switchToSelectiveSearchQaulity()

# run selective search segmentation on input image
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))
```

位于 contrib 模块中
