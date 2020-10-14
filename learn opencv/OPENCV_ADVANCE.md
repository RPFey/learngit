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
