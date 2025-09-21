from __future__ import annotations
import cv2
import numpy as np
from rich import print
from utils.opencv_utils import putBText
from scipy.spatial.transform import Rotation
from scipy import optimize
from enum import Enum
from utils.utils import boundary


class Vision:
    
    def __init__(self, camera_matrix, dist_coeffs, cam_config) -> None:

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.cam_config = cam_config
        self.lower_red_1 = np.array([0,   50, 50])   # Adjust as needed
        self.upper_red_1 = np.array([10, 255, 255])  # Adjust as needed

        self.lower_red_2 = np.array([170, 50, 50])   # Adjust as needed
        self.upper_red_2 = np.array([180, 255, 255]) # Adjust as needed

            # Blue color range
        self.lower_blue = np.array([100, 150, 50]) 
        self.upper_blue = np.array([140, 255, 255])


    def rotation_matrix(self,x_rotation, y_rotation, z_rotation):
        alpha = z_rotation
        beta = y_rotation
        gamma = x_rotation

        # https://en.wikipedia.org/wiki/Rotation_matrix
        # rotates around z, then y, then x

        return np.array([
            [np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma)],
            [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)],
            [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]
        ])

    def vectors_to_transformation_matrix(self,rotation, translation):
    ### Your code here ###
        T = np.eye(4)
        T[:3,:3] = rotation
        T[:3,3] = translation
            
            ###
        return T
    def transformation_matrix_to_vectors(self,t):
        x_rotation = t[:3,0]
        y_rotation = t[:3,1]
        z_rotation = t[:3,2]
        translation = t[:3,3]
        return np.array([x_rotation,y_rotation,z_rotation]), translation
    
    def find_block_angle_and_offset(
        image,
        lower_color=None,
        upper_color=None,
        min_area=5000,
        camera_fov_deg=60
    ):
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove small noise
        
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
            return None,None,mask
        
        # Compute centroid of the largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None, None, mask
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        height, width = image.shape[:2]
        center_x = width // 2
        x, y, w, h = cv2.boundingRect(largest_contour)
        # bottom-center pixel
        bottom_center_x = x + w // 2
        bottom_center_y = y + h
        horizontal_offset = cx - center_x
        fraction_of_half_image = float(horizontal_offset) / (width / 2.0)
        angle_to_block = fraction_of_half_image * (camera_fov_deg / 2.0)
        
        return angle_to_block, (bottom_center_x, bottom_center_y), mask
    



    def detections(self, img: np.ndarray, draw_img:np.ndarray, robot_pose: tuple, kind: str = "aruco") -> tuple:
        
        # PUT YOUR CODE HERE.
        marker_size = 0.05

        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

        marker_corners, ids, rejectedCandidates = detector.detectMarkers(img)
        if ids is None:
            ids = []
        else:
            ids = ids.flatten()
        marker_rvecs = []
        marker_tvecs = []

        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]])
       
        for i, corners in enumerate(marker_corners):
            imagePoints = corners.reshape(-1,2)
            sucess, rvec, tvec = cv2.solvePnP(marker_points,imagePoints,self.camera_matrix,self.dist_coeffs)
            if sucess:
                marker_rvecs.append(rvec)
                marker_tvecs.append(tvec)

        x_offset_camera = 0
        y_offset_camera = 0
        z_offset_camera = 0.295

        x_angle_camera = -126.5
        y_angle_camera = 0
        z_angle_camera = 0

        rotation_camera_robot = np.array([np.radians(x_angle_camera),np.radians(y_angle_camera),np.radians(z_angle_camera)])
        offset_camera_robot = np.array([x_offset_camera,y_offset_camera,z_offset_camera])

        rotation_mat = self.rotation_matrix(rotation_camera_robot[0],rotation_camera_robot[1],rotation_camera_robot[2])
        translation_vector = offset_camera_robot
        T_camera_robot = self.vectors_to_transformation_matrix(rotation=rotation_mat,translation=translation_vector)

        marker_rvecs_robot = []
        marker_tvecs_robot = []

        landmark_rs = []
        landmark_alphas = []
        for rvec, tvec in zip(marker_rvecs, marker_tvecs):
            rotation_mat_marker, _ = cv2.Rodrigues(rvec)

            transformation_marker_camera = self.vectors_to_transformation_matrix(rotation_mat_marker, np.array(tvec).flatten())

            transformation_marker_robot = T_camera_robot @ transformation_marker_camera
            rvec_robot, tvec_robot = self.transformation_matrix_to_vectors(transformation_marker_robot)
            marker_rvecs_robot.append(rvec_robot)
            marker_tvecs_robot.append(tvec_robot)
            x = tvec_robot[0]
            x = -x      # flipped x to solve viewer problem
            y = tvec_robot[1]
            z = tvec_robot[2]
            distance = np.sqrt(x*x+y*y)
            angle_to_marker = np.arctan2(x, y)

            landmark_alphas.append(angle_to_marker)
            landmark_rs.append(distance)

        landmark_positions = marker_tvecs_robot
        #print(f"vision: {landmark_positions}")
        #print(f"alphas: {landmark_alphas[0]}")
        #print(f"radius: {landmark_rs[0]}")
        return ids, landmark_rs, landmark_alphas, landmark_positions


