import cv2
import numpy as np
import os

hsv_ranges = {
    'banana': {
        'ripe': ((20, 100, 100), (35, 255, 255)),    
        'unripe': ((35, 50, 50), (85, 255, 255))    
    },
    'tomato': {
        'ripe': ((0, 100, 100), (10, 255, 255)),     
        'ripe2': ((160, 100, 100), (180, 255, 255)), 
        'unripe': ((35, 50, 50), (85, 255, 255))    
    }
}

def classify_fruit(image_path, fruit_name):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}. Check path!")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if fruit_name == 'tomato': 
        ripe_mask1 = cv2.inRange(hsv, np.array(hsv_ranges['tomato']['ripe'][0]), np.array(hsv_ranges['tomato']['ripe'][1]))
        ripe_mask2 = cv2.inRange(hsv, np.array(hsv_ranges['tomato']['ripe2'][0]), np.array(hsv_ranges['tomato']['ripe2'][1]))
        ripe_mask = cv2.bitwise_or(ripe_mask1, ripe_mask2)
    else:
        ripe_lower, ripe_upper = hsv_ranges[fruit_name]['ripe']
        ripe_mask = cv2.inRange(hsv, np.array(ripe_lower), np.array(ripe_upper))

    unripe_lower, unripe_upper = hsv_ranges[fruit_name]['unripe']
    unripe_mask = cv2.inRange(hsv, np.array(unripe_lower), np.array(unripe_upper))

    ripe_pixels = cv2.countNonZero(ripe_mask)
    unripe_pixels = cv2.countNonZero(unripe_mask)

    classification = 'ripe' if ripe_pixels > unripe_pixels else 'unripe'

    print(f"{fruit_name} in {image_path} is {classification}")

    cv2.imshow(f'{fruit_name} Original', img)
    cv2.imshow(f'{fruit_name} Ripe Mask', ripe_mask)
    cv2.imshow(f'{fruit_name} Unripe Mask', unripe_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return classification

def classify_folder(folder_path, fruit_name):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not files:
        print("No images found in the folder!")
        return

    for file in files:
        image_path = os.path.join(folder_path, file)
        classify_fruit(image_path, fruit_name)

if __name__ == "__main__":

    banana_folder = r"D:\HCL-IP\Assignment-1\Color_based_thresh\bananas"
    tomato_folder = r"D:\HCL-IP\Assignment-1\Color_based_thresh\tomatoes"


    classify_folder(tomato_folder, 'tomato')
