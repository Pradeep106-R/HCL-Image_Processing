import cv2
import numpy as np

img = cv2.imread("bottle.png")

if img is None:
    raise FileNotFoundError("Image not found. Check the filename and path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,            
    minDist=20,        
    param1=50,          
    param2=30,        
    minRadius=5,       
    maxRadius=50        
)

output = img.copy()

if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        cv2.circle(output, (x, y), r, (0, 0, 255), 2)   
        cv2.circle(output, (x, y), 2, (0, 255, 0), 3)  
else:
    print("No bubble-like defects detected.")


cv2.imshow("Original", img)
cv2.imshow("Detected Bubble-like Defects", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output_bubbles_detected.jpg", output)
