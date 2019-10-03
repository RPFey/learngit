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



# SVM

save 模型时 ， 有 process finished with exit code 139 

这是因为写地址溢出了，只需要把 .xml 删掉再写就可以了

关于装载模型的问题， 

```python
svm = cv2.ml.SVM_load(cv2.samples.findFile('./svm.xml'))
```

这里采取 与 haar cascade 相同的装载方法，且要求版本为 4.1 以上。







