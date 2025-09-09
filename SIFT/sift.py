import cv2
import numpy as np

try:
    image = cv2.imread('hazelnut.png')
    if image is None:
        raise FileNotFoundError("Image not found. Please check the file path.")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('Hazelnut with SIFT Keypoints', image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('hazelnut_with_defect_sift.png', image_with_keypoints)
    print(f"Detected {len(keypoints)} keypoints.")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")