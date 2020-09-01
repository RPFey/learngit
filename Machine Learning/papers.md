# Dataset

## [SUN3D](http://vision.princeton.edu/projects/2013/SUN3D/paper.pdf)

### Introduction

<font color='#ff0000'>incorporate 3D reconstruction with 3D labeling, since you just need to label a single object, and after reconstructio, you can get all the labels in all the frames. </font> 

Some resources:

1. RGB-D camera to prodeuce 3D models of spaces [10, 4, 15]  
2. Neuro Science [17] study finds that PPA(represent scenes and spatial layouts, has a view-specific representation ) and RSC (for navigation, route learning , has a place-centric representation ) are complementary.
3. Other 3D dataset : NYU Depth Dataset \ SUN[24]

### Capturing Process

this dataset can be viewed as **Point Cloud** , **3D object model**, **viewpoints of objects**, **bird's-eye view of room**.

Setup: sensor: ASUS Xtion PRO LIVE

Register Methods: We begin by matching key-points using SIFT and remove poor matches using the ratio test. With SIFT results, <font color='#ff0000'> we choose the **ones with valid depth values** </font>, and use a 3-point-algorithm inside a RANSAC loop to find relative transformation. As for fillng te holes in depthm map, one approach is to use cross-bilateral [20] , OR TSDF(voxelize the space) [15].

Loop closure: Bag of Words ; Joint 2D , 3D bundle adjustment :
$$ min \sum_{c}\sum_{c \in V(c)}(||\tilde{x_{p}^{c}}-K[R_{c}|t_{c}]X_{p}||^{2} + \lambda ||\tilde{X_{p}^{c}} - [R_{c}|t_{c}X_{p}||^{2}) $$
$\tilde{X_{p}^{c}}$ and $\tilde{x_{p}^{c}}$ is the observed 3D(RGB-D), 2D point and $X_{p}$ is the ground truth 3D point. (bundle adjustment to minimize the error)

### implementation

github 上 SUN3D 官方的 sfm 中(已经 fork) 指出了深度预先处理。这里注意深度图是 16-bit，所以读出图像应该采用

```python
depth = cv2.imread(img_name, cv2.ANYDEPTH)
```

使用官方的 cpp 库下载数据时，需要安装 png++ 库。方法如下

```bash
sudo apt install zlib1g-dev libpng12-dev
wget -c http://download.savannah.nongnu.org/releases/pngpp/png++-0.2.9.tar.gz
tar -xzf png++-0.2.9.tar.gz
cd png++-0.2.9/
make
make test
sudo make install
```

关于数据集中相邻帧之间的变换关系，已经在文件夹下的 SUN3D.py 中展示。

## Jacquard: A Large Scale Dataset for robotic Grasp Detection

### Introduction

SOTA methods include predicting the manually labeled grasp or evaluating the previuosly generated candidates and select the best one.

Summary:

1. Jacquard synthetic Dataset
2. SGT and other criterion for grasp evaluation

### Model Grasp

representation of a grasp is :
$$ g = {x,y,h,w,\theta} $$
This coordinate can be easily expressed in image coordinate system. As shown in the figure:<br />
<img src=../img/CVpaper/JacquardGrasp.png height="300" wdth="400" /><br/>
z position and approach can be obtained through depth image.

### Generate synthetic dataset

generate scene description: select model, resize them, add mass and drop into the scene. 

Image rendering: render two more RGB images with a projected pattern and applied a stereo-vision algorithm [19] to them. (add noise)

Annotation Generation: generate thousands of potential grasps and perform trials in simulation. The distribution for grasp generation is concentrated around the aligned edges.

### Model Assessment

grasp criterion in Cornell Dataset (rectangle metrics):

1. angle between prediction and ground-truth is below a certain threshold.
2. IoU is over a threshold.

SGT (simulated grasp trial-based criterion) : rebuild the corresponding scene in the simulation environmnt to test whether a predicted grasp is good or not. (This may be the final test metrics)

# Object Detection

## FCOS: Fully Convolutional One-Stage Object Detection

Based on the FCN (fully convolutional network) framework. Other works are [20,28,9,19].

### Arch

For a feature map $F_{i} \in R^{H*W*C}$, any point $M_{i} = F(x,y)$ on the feature map (x,y) maps back to point $I(x,y) = ([\frac{s}{2}]+xs, [\frac{s}{2}]+ys)$ on original image. s is the total stride to this feature map. **Then directly regress the target at the location.** as [20].

If the point $M_{x,y}$ with its original point $I_{x,y}$ falls into the ground-truth box $B_{i} = (x_{0}, y_{0}, x_{1}, y_{1}, c)$ , it's considered as a positive sample. Otherwise, it's a negative sample. The regress target is coded as $\mathbf{\mathop{t^{*}}} = (l^{*}, t^{*}, r^{*}, b^{*})$. Four parametes are the distance between the point to the four sides of the ground truth box respectively. **If a point is in multiple ground-truth boxes, we choose the minimul area box as its ground-truth box**.
> The positive samples are more than anchor-based method.<br />
> This method is like predicting bounding boxes on the segmentation prior.

Loss function : <br />
classification : focal loss in [15] <br/>
regression : IOU loss in [32]<br/>

**Incorporate Features at Multiple Levels**: 在不同尺度的特征图下检测，网络结构如下图:<br/>
<img src=../img/CVpaper/FCOS_Arch.png height="300" width="400"/><br/>
{P3, P4, P5, P6, P7} are feature levels with stride 8,16,32,64,128。在不同尺度上限制一个范围$(m_{i-1}, m_{i})$，当预测的 bbox 超过这个范围则舍弃，作为负样本。

**Center-ness Loss**: supress the low-quality detected bounding boxes. (those far away from the center of a ground truth box)

As shown in the architecture figure, a centerness branch is added in parallel with the classification branch to predict the centerness of the predicted box. The target center-ness for the box is defined as follow:
$$ centerness = \sqrt{\frac{min(l^{*}, r^{*})}{max(l^{*}, r^{*})} * \frac{min(t^{*}, b^{*})}{max(t^{*}, b^{*})}} $$
$\{l^{*}, r^{*}, t^{*}, b^{*}\}$ is the distance to the ground truth box it belongs to. 0 --> corner of the box ; 1 --> center of the box. Use a binary cross entropy loss for it. This center loss can surpress those points at the corner of the bounding boxes. 
<img src=../img/CVpaper/Centerness.png height="300" width="400" />
In the figure above, after multiplying center-ness on the classification score, high classification score with low IOU points are surpressed.
> this branch can also be added in parrallel with the regression block.

### Experiment

FCOS can replace RPN module in some networks.

### Arguments

检测是基于分割的，在每个点上预测目标的位置，虽然不需要anchor box, 但是产生的 bbox 确实有些多。

后面的 center-ness score 作为一种抑制边缘点的方法，能通过 threshold 或者 NMS 有效去除在物体边缘的预测点。减少 bbox 。

看了文章的(More discussion)之后，有些感想。一个是目前整合是一个方向，希望把所有的模块整合到卷积神经网络之中。本文的 center-ness 在另一篇文章中是一个独立的模块，这里将它整合到网络中。

# Monocular - 3D vision

## UnOS: Unified Unsupervised Optical-flow and Stereo-depth Estimation by Watching videos

# Robots

## DroNet: Learning by Flying

### Structure

the first one --> steering angle is a regression, and the second --> collision probablity is an estimate of the uncertainty of the "steering angle". (that's why in Beauty and the Beast, they use the network to predict the variance of the Kalman filter input)

Considering the huge difference of the gradient from regression and classification, the loss:
$$ L_{total} = L_{MSE} + max(0,1-exp^{-decay(epoch-epoch_{0})})L_{BCE} $$
for proportion for classification increases along training.

### Experiment

Collecting the collision dataset: manually label the collected images as 0 (non-collision) and 1 (collision)

Test on DataSet : They use root-mean-squared error(RMSE) and explained variance ratio(EVA) to assess performance on collision prediction.

Real World Navigation: Test the model to fly through urban paths. The flying policy --> Minimize Probability of Collision Policy: choose the direction that minimizes the collision probability. Algorithm[10].

other extensions: [25] 3D-trajectory generation ; [26] estimation of distance;

### thoughts & cites

the reason to use resnet is because the resnet scheme is proposed to address the degradation problem generated by difficulties in network optimization.[18] This really enhance the performance and speed of the network.

problems caused by the two branch. The gradient of classification and regression varies, **so the weight for this two loss must balance the inherent difference of their gradient**.

paper [24] is used to explain the attention of the network, where does the drone focus on? <font color='#ff0000'> the author finds the drone relies on line-like features. (this may not suit our application)</font>

## Real-time 3D Reconstruction on Construction site using Visual SLAM and UAV

### Intro

photogrammetry has advantages like texture rich, low-cost and light weighted. Although the noise floor of image based 3D model is nearly three times higher that Lidar, the density points is much larger, which shows a better representation.

General framework for visual slam: Visual simultaneous localization and mapping: a survey ; "Appearance-based loop closure detection for online large-scale and long-term operation" & "Online global loop closure detection for large-scale multi-session graph-based slam"

<a id="FocalLoss"></a>

#### Focal Loss

Focal Loss 是在原有的交叉熵函数上改进。交叉熵为：
$$ L = -ylogy'-(1-y)log(1-y') $$
Focal Loss:
$$ L_{fl} = \left\{
\begin{aligned}
-\alpha(1-y')^{\gamma}logy' & \quad y=1 \\
-y'^{\gamma}log(1-y') & \quad y=0 \end{aligned} \right. $$
调节$\gamma$使函数对错分的样本的损失函数值更大，模型更加关注错分的样本。而正确分类的样本损失值会变小。平衡因子$\alpha$ 平衡正负样本之间的数量。避免由于负样本占主导。
