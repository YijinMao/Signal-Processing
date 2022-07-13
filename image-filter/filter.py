import cv2,skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 600  #图片像素
plt.rcParams['figure.dpi'] = 150  #分辨率


# 原图
srcImg = cv2.imread("test1.png")
cv2.imshow("src image", srcImg)

# 灰度图
grayImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)

# 给图像增加高斯噪声
noiseImg = skimage.util.random_noise(srcImg, mode='s&p')
cv2.imshow("image with noise", noiseImg)

# ========================================================
# 邻域滤波
# ========================================================
# 方框滤波
boxImg = cv2.boxFilter(noiseImg, ddepth = -1, ksize = (2, 2), normalize = False)
cv2.imshow("box Image", boxImg)

# 均值滤波
blurImg = cv2.blur(noiseImg, (6, 5))
cv2.imshow("blur image", blurImg)

# 高斯滤波
gaussImg = cv2.GaussianBlur(noiseImg, (5, 5), 0)
cv2.namedWindow("gaussain image")
cv2.imshow("gaussain image", gaussImg)

# 中值滤波
medImg = cv2.medianBlur(np.uint8(noiseImg * 255), 3)
cv2.namedWindow("median image")
cv2.imshow("median image", medImg)

# ========================================================
# 频域滤波
# ========================================================
# 傅里叶变换
dft = cv2.dft(np.float32(grayImg), flags = cv2.DFT_COMPLEX_OUTPUT)
# 将图像中的低频部分移动到图像的中心
dftShift = np.fft.fftshift(dft)
# 计算幅频特性
magnitude = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))

# 定义滤波掩码
def mask(img, ftype):
    crow, ccol = int(img.shape[0] / 2), int(img.shape[1] / 2) # 求得图像的中心点位置
    # 低通
    if ftype == 'low':
        mask = np.zeros((img.shape[0], img.shape[1], 2), np.uint8)
        mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    # 高通
    if ftype == 'high':
        mask = np.ones((img.shape[0], img.shape[1], 2), np.uint8)
        mask[crow-30:crow+30, ccol-30:ccol+30] = 0
    return mask

highImg = dftShift * mask(grayImg, 'high')
highImg = np.fft.ifftshift(highImg)
highImg = cv2.idft(highImg)
highImg = cv2.magnitude(highImg[:, :, 0], highImg[:, :, 1])

# plt.subplot(121), plt.imshow(grayImg, cmap = 'gray')
# plt.title('原图'), plt.xticks([]), plt.yticks([])
plt.plot(), plt.imshow(magnitude, cmap = 'gray')
plt.title('频谱图'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(highImg, cmap = 'gray')
# plt.title('高通滤波图'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
