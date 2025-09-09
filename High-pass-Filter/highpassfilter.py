import cv2
import matplotlib.pyplot as plt

image = cv2.imread('bottle.png') 
if image is None:
    raise FileNotFoundError("Could not load image. Check the path!")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

laplacian_abs = cv2.convertScaleAbs(laplacian)

plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1)
plt.title("Original (Gray)")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("High-Pass (Laplacian)")
plt.imshow(laplacian_abs, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

cv2.imwrite('high_pass_laplacian.jpg', laplacian_abs)
