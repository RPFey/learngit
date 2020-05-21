# Object Detection

## positive samples and negative samples selection

* 正负样本定义

1. 首先初始化每个 anchor 的 mask 都是 -1 ，表示都忽略 anchor 。
2. 将每个 anchor 和所有的 gt 的 iou 的最大 IoU 小于 neg_iou_thr 的 anchor 的 mask 设置为 0 ，表示为负样本。
3. 对于每个 anchor ，计算其和所有 gt 的 IoU ，选取最大的 IoU 对应的 gt 位置，如果其最大 IoU 大于等于 pos_iou_thr ，则设置该 anchor 的 mask 为 1, 表示该 anchor 负责预测该 gt bbox　，是高质量的 anchor
4. 3 的设置会使得某些 gt 没有分配到对应的 anchor , 对这些 gt 找出最大 IoU 的 anchor 位置，如果其 IoU 大于 min_pos_iou 则设置其为对应 anchor , 否则忽略。

对负样本的采样存在困难例(hard example)与简单例(simple example)之间比例不平衡，在Libra R-CNN 中提出 IoU-balanced Sampling :

对负样本按照 IoU 划分 k 个区间，每个区间再进行随机采样，保证易学习负样本和难负样本比例尽量均衡。

实际代码是先在 k 个区间中均匀采样，如果不足就全部采样。之后在其余区间中再次采样以补足缺少的样本。

* 设计损失函数

回归的损失一般会大于分类的损失。回归中的 outlier 会造成巨大的梯度不平衡。平衡回归目的是减小外点对梯度的影响，同时突出内点中难易样本的梯度差异。由 smooth L1 改进，

$$
L_{loc} = \sum_{i \in {x,y,w,h}} L_{b}(t_{i}^{u} - v_{i}) 
$$

其梯度为

$$
\frac{\partial L_{loc}}{ \partial w} \propto \frac{\partial L_{b}}{\partial t_{i}^{u}} \propto \frac{\partial L_{b}}{\partial x}
$$

并设计梯度为

$$
\begin{equation}
\frac{\partial L_{b}}{\partial x} = \left\{
\begin{aligned}
& \alpha ln(b|x| + 1) & if \quad |x|<1 \\
& \gamma & otherwise,
\end{aligned}
\right.
\end{equation}
$$

这里利用了 ln(x) 是上凸函数的性质。因此原函数为

$$
\begin{equation}
L_{b} = \left\{
\begin{aligned}
& \frac{\alpha}{b} (b|x| + 1)ln(b|x| + 1) - \alpha |x| & if \quad |x|<1 \\
& \gamma |x| + C & otherwise,
\end{aligned}
\right.
\end{equation}
$$

为了函数的连续性，需要保持

$$
\alpha ln(b+1) = \gamma
$$

<img src=../img/CVpaper/BalanceLoss.jpg />

可以看出，在 [0,1] 区间上，平衡后的梯度比 Smoothl1 更具有区分度。

## Rotated IoU

facebook detectron2 has implemented rotated IoU in CUDA.
