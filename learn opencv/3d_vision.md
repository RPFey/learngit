
<!-- vim-markdown-toc GFM -->

* [3D Vision](#3d-vision)
	* [Depth image preprocess](#depth-image-preprocess)
		* [Adaptive Noise Reduction Method Using Noise Modeling for TOF sensor](#adaptive-noise-reduction-method-using-noise-modeling-for-tof-sensor)
	* [3D Reconstruction](#3d-reconstruction)
	* [OpenCV](#opencv)
		* [triangulatePoints](#triangulatepoints)
	* [Python module](#python-module)
		* [Open3d](#open3d)

<!-- vim-markdown-toc -->
# 3D Vision

## Depth image preprocess

### Adaptive Noise Reduction Method Using Noise Modeling for TOF sensor

Adaptive Noise Reduction Method Using Noise Modeling For ToF Sensor

ToF sensor noise: scattering of IR light, moving object in the scene, low reflective object, exceeding the capacity (which results 0 in the image, it's called saturation region). papers about ToF camera are [1,2]

to recover the saturation region, 8 neighbors are selected:
$$ D_{sat} = \frac{1}{N}\sum_{i=0}^{N}D_{i} $$
$D_{sat}$ is the restored saturation pixel. <font color='#ff0000'> this is effective when the saturation region is small. </font>
> For larger region, the operation can continue when all the pixels are non-zero

depth noise can be estimated from depth variation. They model the relation between amplitude and distance using $a_{i}$, $b_{i}$, $c_{i}$ in [6]. Images are taken from 1.0 m to 4.0 m with 0.5 m interval.

After we get the model function $f$, the intensity parameter $\sigma_{r}$ for color smoothing is deicided by $\sigma_{r}=f(a_{i}, d_{i})$, $a_{i}$ is the amplitude, $d_{i}$ is the distance value.

## 3D Reconstruction

### Low Cost 3D surface reconstruction using stereo camera for small object

exsiting methods in 3D reconstruction. [6,7] Point cloud construction [8]. Quality analysis [9]. Capture and project the texture [10]. Compared to camera images, laser beam provide more details of the surface of the objects.[12] Texture mapping to recover the texture. [13]

## OpenCV

### triangulatePoints

三角法测距

```c++
cv::Mat pts_4d;
cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);
for (int i = 0; i < pts_4d.cols; i++) {
    cv::Mat x = pts_4d.col(i);
    x /= x.at<float>(3, 0); // 归一化
    cv::Point3d p(
        x.at<float>(0, 0),
        x.at<float>(1, 0),
        x.at<float>(2, 0)
    );
    points.push_back(p);
}
```

注意处理方法，pts_4d 是一个四维的其次坐标点，采用Mat 类型来接受。

## Python module

### Open3d

A python module to load and save point cloud.
