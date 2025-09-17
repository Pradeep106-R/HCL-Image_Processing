import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "image1.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 100, 200)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

open_close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.title("Original Edge Mask")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("After Opening")
plt.imshow(opening, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("After Closing")
plt.imshow(closing, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Opening + Closing")
plt.imshow(open_close, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
