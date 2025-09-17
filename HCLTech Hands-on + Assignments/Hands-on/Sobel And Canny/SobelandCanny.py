import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_sobel(image):
 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    magnitude = cv2.magnitude(sobelx, sobely)

    scale_factor = np.max(magnitude) / 255
    magnitude = (magnitude / scale_factor).astype(np.uint8)
    
    return magnitude

def apply_canny(image, low_threshold=100, high_threshold=200):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    return edges

def plot_edges(original, sobel_edges, canny_edges, title):

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(sobel_edges, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.show()

image_path = 'image1.png' 
image = cv2.imread(image_path)

if image is not None:

    sobel_edges = apply_sobel(image)
    canny_edges = apply_canny(image)

    plot_edges(image, sobel_edges, canny_edges, 'Edge Detection on Industrial Image')
else:
    print(f"Error: Unable to load image at {image_path}")