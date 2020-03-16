## distance

4-adjacency:
$$D_{4}(p,q) = |x-s| + |y-t| $$
eg:

			2
		2	1	2
	2	1	0	1	2
		2	1	2	
			2

8-adjacency:
$$D_{8}(p,q) = max(|x-s|,|y-t|) $$

	1	1	1
	1	0	1
	1	1	1

m-adjacency:

			p3 	p4
	p1	p2	
	p	

p - p4 距离由之间的像素点决定，若 p1, p3 为0，则距离是2. 若不为0， 则距离是4.

# Image Enhancement

## intro

spatial domain and frequecy domain

spatial domain
$$g(x,y) = T[f(x,y)]$$

f input image; g processed image. A window method is applied

<div align="center"><img src=../img/digital_img/Window.jpg width="500" height="300" /></div>

single pixel in gray level 
$$ s = T(r) \quad r \quad in \quad f(x,y) ; \quad s \quad in \quad g(x,y) $$
multiple pixels : mask , kernel, window ...

## Gray level transformation

<img src=../img/digital_img/gray_level.png width="500" height="400" align="center"/>

log transform expand darker pixels && compress the dynamic range in pixel values
$$ s = clog(1+r) $$
可以认为 log 变换将低频的信号`拓展`，使得在 Fourier 变换后可以处理低频的信息。由于傅立叶变换后，值在$ 0 ~ 1.5*10^{6} $ , 如果单纯采用线性变换，那么最大值会尤为突出，而较小范围的值则很难分辨。

$$ s = c(r + \epsilon) ^{\gamma} $$
$\gamma < 1$ 效果等同于 log transform, $\gamma > 1$ 则是将较大值的区域`拓展`。*power tranform* applies in many devices.Like in CRT, intensity-to-voltage is a power function.*Gamma correction* is needed when we want to explore details in darker(low) value area.For example:
$$ in \quad CRT :  \qquad s = r^{2.5} $$
$$ Correction : \qquad s = r^{\frac{1}{2.5}}=r^{0.4} $$

Piecewise-Linear Function is a more versatile method.By designing th eprameters, we can stretch the gray levels of the image. 
<div align="center"><img src=../img/digital_img/PiesewiseLinear.png width="400" height="380"/></div>

Gray-level slicing : Focus on a certain interval of pixel intensities.
<div align="center"><img src=../img/digital_img/Slicing.png width="500" height="360" /></div>
For the left image, the texture of other region is discarded. For the right one, it's preserved.

Bit-plane slicing : For instance, a pixel contains 8-bit information for its gray level. The higher-order bits contain the majority of visually significant data, while other bits contain subtle details. Seperating      the pixel data into its bit planes is useful to analyze the importance of each bit. Compression is applied to ignore those unsignificant bits.

<font face='DejaVu Sans Mono' size='3'>
## Histogram

A histogram is a function:
$$ h(r_{k}) = n_{k} $$
$n_{k}$ is the number of pixels in gray level $r_{k}$. A normalized hisogram is :
$$ p(r_{k}) = \frac{n_{k}}{n} \quad for \quad k = 0,1,...L-1 $$
From the histogram, we can find the distribution of gray levels. A narrow histogram indicates a low contrast image while a wide one indicates high-contrast.

Consider the transformation
$$ s = T(r) $$

1. $T(r)$ is **single-value** and **monotonically increasing** in interval
2. $0\leq T(r) \leq 1 \qquad for \quad 0\leq r \leq 1$

Gray Levels can be viewed as random variables in interval [0,1]. Let $p_{r}(r)$ and $p_{s}(s)$ denote p.d.f. The relation is :
$$ p_{s}(s) = p_{r}(r)|\frac{dr}{ds}| $$
the variable, s, is determined by the chosen transformation function. A useful one is:
$$ s=T(r)=\int_{0}^{r}p_{r}(w)dw $$
(this is just the cumulation function) since:
$$ \begin{aligned} \frac{ds}{dr} &= \frac{dT(r)}{dr}　\\
   								 &= \frac{d}{dr}[\int_{0}^{r}p_{r}(w)dw] \\
   								 &= p_{r}(r) \end{aligned} $$
thus (from the probability theory, the distribution of $p_{s}(s)$ is):
$$ \begin{aligned} p_{s}(s) &= p_{r}(r)|\frac{dr}{ds}| \\
     						&= p_{r}(r)|\frac{1}{p_{r}(r)}| \\
     						&= 1 \qquad 0 \leq s \leq 1  \end{aligned}$$
The uniform distribution of $p_{s}(s)$ is independent of $p_{r}(r)$. For discrete values:
$$ p_{r}(r_{k}) = \frac{n_{k}}{n} \qquad k = 0,1,2,...,L-1 $$
$$ s_{k} = \sum_{j=0}^{k} \frac{n_{j}}{n} \qquad k = 0,1,2,...,L-1 $$
The equation above is called **histogram equalization**. *why?Because the distribution is uniformly distributed!*

</font>
gamma transform and histogram normalization. Gamma transform has to adjust the parameter, while histogram normalization dosen't.

If given a specified histogram $z_{q}$, we want to transform an input image $s_{k}$. We perform histogram normmalization oon both images and match the corresponding buckets.We normalize the histogram of the original image by the transform below:
$$ s = T(r) = (L-1)\int_{0}^{r}p_{r}(w)dw $$
Define the variable z of the target image:
$$ G(z) = (L-1)\int_{0}^{z}p_{z}(t)dt = s $$
the relation between two images are:
$$ z = G^{-1}(s) = G^{-1}[T(r)] $$
>Notice : the normalization is pixel-wise

Histogram operation can be performed locally, just in a window case. This can enhance the local texture.
$$ \sigma^{2}_{S_{xy}} = \sum_{i=0}^{L-1}(r_{i}-m_{S_{xy}})^{2} $$
<src img />

## spatial filter

A filter consists of a neighborhood and a predefined operation
$$ g(x,y) = \sum_{s=-a}^{a}\sum_{k=-b}^{b}T(f(s,k)) $$

