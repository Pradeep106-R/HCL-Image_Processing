import cv2

gray = cv2.imread('transistor.png', cv2.IMREAD_GRAYSCALE)

thresh_value = 127
max_value = 255
_, thresh_img = cv2.threshold(gray, thresh_value, max_value, cv2.THRESH_BINARY)

cv2.imshow('Thresholded Image', thresh_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('thresholded_image.jpg', thresh_img)

img = cv2.imread('transistor.png')
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
