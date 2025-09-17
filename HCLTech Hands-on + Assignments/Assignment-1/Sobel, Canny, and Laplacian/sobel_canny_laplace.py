import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("000.png", cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(img, (5, 5), 0)

sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.convertScaleAbs(sobel)

canny = cv2.Canny(blurred, 80, 180) 

laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)

plt.figure(figsize=(14,8))

plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(2,2,2)
plt.title("Sobel Edges")
plt.imshow(sobel, cmap='gray')
plt.axis("off")

plt.subplot(2,2,3)
plt.title("Canny Edges")
plt.imshow(canny, cmap='gray')
plt.axis("off")

plt.subplot(2,2,4)
plt.title("Laplacian Edges")
plt.imshow(laplacian, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
