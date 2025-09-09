import cv2
import matplotlib.pyplot as plt
original_bgr = cv2.imread('hazelnut.png')
if original_bgr is None:
    print("Error: Could not load image. Please check the file path.")
else:
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    original_grayscale = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

    filtered_image = cv2.blur(original_grayscale, (5, 5))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_rgb)
    axes[0].set_title('Original (RGB)')
    axes[0].axis('off') 

    axes[1].imshow(original_grayscale, cmap='gray')
    axes[1].set_title('Original (Grayscale)')
    axes[1].axis('off')

    axes[2].imshow(filtered_image, cmap='gray')
    axes[2].set_title('Mean Filtered (Grayscale)')
    axes[2].axis('off')

    plt.tight_layout() 
    plt.show()

