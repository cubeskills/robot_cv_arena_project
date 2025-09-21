import numpy as np
from timeit import default_timer as timer 

import cv2
import sys 
import os 
sys.path.insert(1,os.path.join(os.getcwd(),'..'))
from utils.utils import load_config 
from utils.robot_controller import RobotController


config = load_config("config.yaml")
robot=RobotController(config)
robot.__enter__()

def find_block_angle_and_offset(
    image,
    lower_color=None,
    upper_color=None,
    min_area=500,
    camera_fov_deg=60,
):
    """
    Finds a colored block in the image and returns:
      - angle_to_block (in degrees) relative to image center
      - centroid (x,y) in the image
      - mask (for debugging/visualization)
    If no valid contour is found, returns None for angle and centroid.
    
    Arguments:
    ----------
    image       : BGR image (numpy array) from OpenCV.
    lower_color : tuple, lower bound in HSV (h, s, v).
    upper_color : tuple, upper bound in HSV (h, s, v).
    min_area    : int, minimum area of the contour to consider valid.
    camera_fov_deg : float, approximate horizontal field of view of your camera in degrees.
    
    Returns:
    --------
    angle_to_block : float or None, angle in degrees relative to image center (negative = left, positive = right).
    centroid       : (cx, cy) or None, centroid in image coordinates.
    mask           : numpy array, the thresholded mask (for debugging).
    """
    # Convert to HSV
    image = ""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Threshold
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Morphological cleanup (optional, can improve noise issues)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove small noise
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, mask
    
    # Pick the largest contour above a certain area
    largest_contour = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area and area > min_area:
            max_area = area
            largest_contour = cnt
    
    if largest_contour is None:
        # No valid contour found
        return None, None, mask
    
    # Compute centroid of the largest contour
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, None, mask
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Compute angle from horizontal offset
    # - The image center is (width // 2, height // 2)
    # - horizontal_offset_in_pixels = (cx - center_x)
    # - angle = (horizontal_offset_in_pixels / (image_width/2)) * (camera_fov_deg/2)
    #   This is a simple linear approximation based on the camera's approximate FOV.
    
    height, width = image.shape[:2]
    center_x = width // 2
    
    horizontal_offset = cx - center_x
    # Convert offset in pixels to fraction of half-image
    fraction_of_half_image = float(horizontal_offset) / (width / 2.0)
    angle_to_block = fraction_of_half_image * (camera_fov_deg / 2.0)
    
    return angle_to_block, (cx, cy), mask

if __name__ == "__main__":
    # Example usage with a test image
    # For red detection in HSV, you might have to use two ranges if the hue wraps around.
    # For blue, it's simpler, e.g. lower_blue = (100, 100, 50), upper_blue = (130, 255, 255)
    
    # We'll just pick a single range for demonstration (blue-ish range):
    lower_blue = (100, 150, 50)
    upper_blue = (140, 255, 255)
    
    # Load an image (or capture from camera)
    # For a real robot, you'd do something like:
    # cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()
    # while ret:
    #     ... processing ...
    #     ret, frame = cap.read()
    # But here, let's just load from file:
    image = cv2.imread("test_image.jpg")  # Replace with a real path
    
    angle, centroid, mask = find_block_angle_and_offset(
        image, 
        lower_color=lower_blue, 
        upper_color=upper_blue, 
        min_area=500,
        camera_fov_deg=60
    )
    
    if angle is not None:
        print(f"Block detected at angle: {angle:.2f} degrees")
        print(f"Block centroid at: {centroid}")
        
        # Draw debug info
        cv2.circle(image, centroid, 5, (0,0,255), -1)
        cv2.putText(image, f"Angle: {angle:.1f}", (centroid[0], centroid[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    else:
        print("No valid block found.")
    
    # Show the results
    cv2.imshow("Original", image)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
