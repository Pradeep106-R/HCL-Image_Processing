import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = "hazelnut.png"   
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

_, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, blockSize=25, C=10)

adaptive_gauss = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, blockSize=25, C=10)


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Otsu Thresholding")
plt.imshow(otsu_thresh, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Adaptive Mean Thresholding")
plt.imshow(adaptive_mean, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Adaptive Gaussian Thresholding")
plt.imshow(adaptive_gauss, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
