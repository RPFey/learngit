# Opencv-Python

将按照《数字图像处理》章节顺序，记录 python 图像处理函数。

## 频率域滤波

```python
# 采用 numpy FFT 变换模块
f = np.fft.fft2(lena)
# shift the DC component to the center (the origin is on the upper left)
fshift = np.fft.fftshift(f)
fshift = fshift*mask
# shift back
fshift = np.fft.ifftshift(fshift)
f = np.fft.ifft2(fshift)
```
