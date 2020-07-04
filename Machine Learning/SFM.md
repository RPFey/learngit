
<!-- vim-markdown-toc GFM -->

* [Sfm](#sfm)
	* [SIFT (Distinctive Image Features from Scale-Invariant Keypoints)](#sift-distinctive-image-features-from-scale-invariant-keypoints)
		* [Arch](#arch)
		* [Argument](#argument)
		* [补充](#补充)
	* [SuperPoint](#superpoint)
		* [Metric](#metric)

<!-- vim-markdown-toc -->
<font face="DejaVu Sans Mono" size="3">

# Sfm

## SIFT (Distinctive Image Features from Scale-Invariant Keypoints)

### Arch

* Detection of Scale-Space Extrema

DoG (difference of Gaussian):

$$
\begin{aligned}
D(x,y,\sigma) &= (G(x,y,k\sigma) - G(x,y,\sigma)) * I(x,y) \\
            &= L (x,y,k\sigma) - L(x,y,\sigma)
\end{aligned}
$$

advantages:

1. easy to compute
2. provide a close approximation to the scale-normalized Laplacian of Gaussian, $\sigma^{2}\triangledown^{2}G$

采用极坐标形式很容易证明：
$$
\sigma\triangledown^{2}G = \frac{\partial G}{\partial \sigma} \approx \frac{G(x,y,k\sigma) - G(x,y,\sigma)}{k\sigma - \sigma}
$$

> Note : maxima and minima of $\sigma^{2}\triangledown^{2}G$ produce the most stable image features. **Tony Lindeberg指出尺度规范化的LoG算子具有真正的尺度不变性。**

* Costruction

1. Divide each octave of scale space into a integer number, s. So the scaling factor is $k=2^{\frac{1}{s}}$. The variance of Gaussian Kernel for the kth scale is $\sigma_{k} = k * \sigma_{initial}$
2. Adjacent image scales are subtracted to produce DoG
3. Once a complete octave has finished, we downsample the gaussian images by a factor of 2. The process repeated.

> Q:之后的 octave 是在之前 octave 顶层（$\sigma = 2$）采样后，继续卷积得到，还是直接由原图像采样后卷积得到。

> A:从一份实现来看，是在 octave 的顶层 scale 降采样之后直接再卷积。但是如果单从金字塔尺度不变的角度考虑，应该是在原图像上采样后卷积。

* Local Extrema Detection

The point which is greater(or less) than the 8-neighbor (in the current scale) and 9-neighbor (in the adjacent scale) will be considered as a local maxima. Using several stages to filter out points will accelerate the process because most of the points will be filtered out in the first stage.

<font color="#ff0000">Unfortunately, there is no minimum spacing of samples that will detect all extrema.</font>

* Accurate Keypoint Detection

Use the second-order Tayler expansion to find the minima:

$$
D(\bold{x}) = D + \frac{\partial D ^{T}}{\partial \bold{x}} \bold{x} + \frac{1}{2} \bold{x^{T}} \frac{\partial ^{2} D}{\partial \bold{x^{2}}} \bold{x}
$$

where D is evaluated at the sample point from the previous stage and $\bold{x} = (x,y,\sigma)^{T}$ is the *offset* from this point.

> Hessian Matrix 采用差分的 3*3 矩阵近似计算

当得到的 offset $\bold{x}$ 偏离大于 0.5 时，说明最小值点更靠近周边的点。在实际选点的时候，要选择另外的点。在原先点上加上这个偏移作为特征点的精确位置信息。

在极值点处的值可以帮助排除不稳定的特征点。

$$
D(\hat{\bold{x}}) = D +\frac{1}{2} \frac{\partial D ^{T}}{\partial \bold{x}} \hat{\bold{x}}
$$

当 $D(\hat{\bold{x}}) \leq 0.03$ 则可以排除掉这一特征点。

* Eliminate Edge Response

A poorly determined peak in the difference-of-Gaussian function will have a large principal curvature across the edge but a small one in the perpendicular direction.

$$
H = 
\begin{bmatrix}
D_{xx}  & D_{xy} \\
D_{xy}  & D_{yy}
\end{bmatrix}
$$

这里是参考了 Harris Corner Detector 的检测方法，由

$$
D(\bold{x}) = D + \frac{\partial D ^{T}}{\partial \bold{x}} \bold{x} + \frac{1}{2} \bold{x^{T}} \frac{\partial ^{2} D}{\partial \bold{x^{2}}} \bold{x}
$$

求导可得

$$
\triangledown D(\bold{x}) = \triangledown D + H \bold{x}
$$

最大梯度与最小梯度在 H 的两个特征向量方向。所以采用特征值之比的方法判断是否在直线上。记$\alpha$ 和 $\beta$ 为两个特征值。

$$
Tr(\bold{H}) = D_{xx} + D_{yy} = \alpha + \beta
$$

$$
Det(\bold{H}) = D_{xx}D_{yy} - (D_{xy})^{2} = \alpha \beta
$$

有

$$
\frac{Tr(\bold{H})^{2}}{Det(\bold{H})} = \frac{(\alpha + \beta)^{2}}{\alpha \beta} = \frac{(r+1)^{2}}{r}
$$

r 为 $\frac{\alpha}{\beta}$。因此直接用此式来判断是否在直线边界处。

* Orientation Assignment

The orientation histogram has 36 bins covering the 360 degree range of orientations. Each sample added to the histogram is weighted by its gradient magnitude and by a Gaussian-weighted circular window with a sigma that is 1.5 times that of the scale of the keypoint.

```
hist = list[bin_nums]
Gaussian_kernel # the Gaussian weight window

for point in neighboring_points 
    m <- magnitude of the gradient of that point
    theta <- orientation of the gradient of that point
    index <- bin index of theta
    Gauss_weight <- Gaussian_kernel[point.x, point.y]
    weight <- Gauss_weight * m
    hist[index] += weight
```

* Local Image Description

Edelman et al.(1997) : 

> These complex neurons respond to a gradient at a particular orientation and spatial frequency, but the location of the gradient on the retina is allowed to shift over a small receptive ﬁeld rather than being precisely localized. 

<img src=../img/CVpaper/SIFT_descriptor.png />

A Gaussian weighting function with $\sigma$ equal to one half of the descriptor window is used to assign a weight to the magnitude of each sample point.(shown by the circle)

### Argument

使用 DoG 之后，先是在 neighbor 中用比较的方法初步筛选。然后用 Hessian 矩阵再次确定极值点的位置。

从 Edelman (1997) 文献中可以看出来，视觉神经对梯度的方向和频率敏感。但是对于位置不那么敏感。
> 可以看出 pool 作用（小区域内的平移不变性）

### 补充

尺度空间(scale space)思想最早是由Iijima于1962年提出的，后经witkin和Koenderink等人的推广逐渐得到关注，在计算机视觉邻域使用广泛。

尺度空间理论的基本思想是：在图像信息处理模型中引入一个被视为尺度的参数，通过连续变化尺度参数获得多尺度下的尺度空间表示序列，对这些序列进行尺度空间主轮廓的提取，并以该主轮廓作为一种特征向量，实现边缘、角点检测和不同分辨率上的特征提取等。

尺度空间方法将传统的单尺度图像信息处理技术纳入尺度不断变化的动态分析框架中，更容易获取图像的本质特征。尺度空间中各尺度图像的模糊程度逐渐变大，能够模拟人在距离目标由近到远时目标在视网膜上的形成过程。

尺度空间满足视觉不变性。该不变性的视觉解释如下：当我们用眼睛观察物体时，一方面当物体所处背景的光照条件变化时，视网膜感知图像的亮度水平和对比度是不同的，因此要求尺度空间算子对图像的分析不受图像的灰度水平和对比度变化的影响，即满足灰度不变性和对比度不变性。另一方面，相对于某一固定坐标系，当观察者和物体之间的相对位置变化时，视网膜所感知的图像的位置、大小、角度和形状是不同的，因此要求尺度空间算子对图像的分析和图像的位置、大小、角度以及仿射变换无关，即满足平移不变性、尺度不变性、欧几里德不变性以及仿射不变性。

## [SuperPoint](https://arxiv.org/abs/1712.07629)

### Experiment

### Argument

这篇文章首先提出了多头的预测。也就是从 VGG head 中分出 prediction head 和 description head。减小了模型的参数。

提出了一种类似的“自监督学习”方法。也就是先在合成的数据集上学习，然后再在真实图片中学习。同时通过前后 H 变换作为监督。

新的方法：可以采用 PointNet 中的 global - local feature aggregation 的方法，将后层的特征向量上采样后与前层的拼接在一起，再对各个点预测。或者是用 feature pyramid 的手法。

### Metric

* Repeatability

### Implementation

* Magic Point

在 magic_point.py 中。用的是 precision 和 recall

| precision | recall |
| :- | :- |
| 预测正确的点除以所有预测点。| 预测正确的点除以所有的正确点。|

```python
precison = tf.reduce_sum(pred * labels) / tf.reduce_sum(pred)
recall = tf.reduce_sum(pred * labels) / tf.reduce_sum(labels)
```

其中输出操作经过了一个 threshold 和 NMS(optional) 
 
### 代码解读

* base_model.py 

\_gpu_tower : 将数据拆分到不同的 GPU 上。net_output 作为输出。训练时在不同 gpu 上构建计算图，并且提取模型参数和反向传播梯度。

</font>
