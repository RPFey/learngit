import cv2 
import numpy as np
import matplotlib.pyplot as plt 

lena = cv2.imread("./lena.bmp", cv2.IMREAD_GRAYSCALE)

mask = np.zeros((512,512))
mask[206:306, 206:306] = 1

h, w = lena.shape
mean = np.mean(lena)
var = np.var(lena)
print(" mean : {},  var: {}".format(mean,var))

# Without double padding (square low pass filter)
f = np.fft.fft2(lena)
fshift = np.fft.fftshift(f)
fshift = fshift*mask
fshift = np.fft.ifftshift(fshift)
f = np.fft.ifft2(fshift)
# magnitude_spectrum = 20*np.log(np.abs(fshift))
f = np.abs(f)
f = f / np.max(f) * 255
f = f.astype(np.uint8)

plt.subplot(221) 
plt.imshow(lena, cmap='gray')
plt.subplot(222)
plt.imshow(f, cmap='gray')

# with double padding (square low pass filter)
padding = np.pad(lena, ((0,h), (0,w)), 'constant')

mask1 = np.zeros((1024,1024))
mask1[412:612, 412:612] = 1

f_ = np.fft.fft2(padding)
fshift = np.fft.fftshift(f_)
fshift_sq = fshift*mask1
fshift_sq = np.fft.ifftshift(fshift_sq)
f_sq = np.fft.ifft2(fshift_sq)
f_sq = np.abs(f_sq)
f_sq = f_sq[:h, :w]
f_sq = f_sq / np.max(f_sq) * 255
f_sq = f_sq.astype(np.uint8)

error = np.sum(np.abs(f-f_sq)) / (h*w)
print("error : ", error)

# circle low pass filter
x = np.arange(0, 1024, step=1)
y = np.arange(0, 1024, step=1)
x_coor, y_coor = np.meshgrid(x,y)
dis = np.sqrt((x_coor - 512)**2 + (y_coor - 512)**2)
mask = np.zeros((2*h, 2*w))
mask[np.where(dis<150)] = 1

mask_out = (mask*255).astype(np.uint8)
cv2.imwrite("mask.png",mask_out)

fshift_cirlce = fshift*mask
fshift_cirlce = np.fft.ifftshift(fshift_cirlce)
f_cir = np.fft.ifft2(fshift_cirlce)
f_cir = np.abs(f_cir)
f_cir = f_cir[:h, :w]
f_cir = f_cir / np.max(f_cir) * 255
f_cir = f_cir.astype(np.uint8)

plt.subplot(223) 
plt.imshow(f_sq, cmap='gray')

plt.subplot(224)
plt.imshow(f_cir, cmap='gray')
plt.show()