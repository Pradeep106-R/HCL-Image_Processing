import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)

_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


_, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, blockSize=25, C=10)

adaptive_gauss = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, blockSize=25, C=10)


plt.figure(figsize=(12,8))

plt.subplot(2,3,1), plt.imshow(img, cmap='gray'), plt.title("Original"), plt.axis("off")
plt.subplot(2,3,2), plt.imshow(global_thresh, cmap='gray'), plt.title("Global (T=127)"), plt.axis("off")
plt.subplot(2,3,3), plt.imshow(otsu_thresh, cmap='gray'), plt.title("Otsu Threshold"), plt.axis("off")
plt.subplot(2,3,4), plt.imshow(adaptive_mean, cmap='gray'), plt.title("Adaptive Mean"), plt.axis("off")
plt.subplot(2,3,5), plt.imshow(adaptive_gauss, cmap='gray'), plt.title("Adaptive Gaussian"), plt.axis("off")

plt.tight_layout()
plt.show()
