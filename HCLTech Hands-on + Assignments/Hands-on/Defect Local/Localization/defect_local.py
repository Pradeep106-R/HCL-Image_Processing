import cv2
import numpy as np
import matplotlib.pyplot as plt

ref = cv2.imread("img_reference.png", cv2.IMREAD_GRAYSCALE)
test = cv2.imread("img_test.png", cv2.IMREAD_GRAYSCALE)

if ref.shape != test.shape:
    test = cv2.resize(test, (ref.shape[1], ref.shape[0]))

def align_images(template, test, warp_mode=cv2.MOTION_AFFINE, iterations=5000, eps=1e-6):
   
    im1 = template.astype(np.float32)
    im2 = test.astype(np.float32)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)

    cc, warp_matrix = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(test, warp_matrix, (template.shape[1], template.shape[0]),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        aligned = cv2.warpAffine(test, warp_matrix, (template.shape[1], template.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned

aligned_test = align_images(ref, test, warp_mode=cv2.MOTION_AFFINE)

diff = cv2.absdiff(ref, aligned_test)

blur = cv2.GaussianBlur(diff, (5,5), 0)
_, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)

kernel = np.ones((3,3), np.uint8)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

dilated = cv2.dilate(opened, kernel, iterations=1)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
annotated = cv2.cvtColor(aligned_test, cv2.COLOR_GRAY2BGR)

for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    if cv2.contourArea(c) > 100:    
        cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,0,255), 2)

plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(diff, cmap='gray')
plt.title("Difference")

plt.subplot(1,3,2)
plt.imshow(dilated, cmap='gray')
plt.title("Binary Mask of Defects")

plt.subplot(1,3,3)
plt.imshow(annotated[:,:,::-1])
plt.title("Detected Defects")

plt.show()