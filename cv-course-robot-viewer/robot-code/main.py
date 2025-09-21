import cv2
import time
import numpy as np
import sys
import jsonpickle
import pickle
from message import Message
from timeit import default_timer as timer
from numba import typed

from utils.robot_controller import RobotController

from publisher import Publisher
from utils.keypress_listener import KeypressListener
from rich import print
from utils.utils import load_config
from utils.utils import PIDController 
from continuous import intersecting, cone_check, check_path, combine_path, Explorer, end_check
from utils.utils import difference_angle

from enum import Enum
class TaskPart(Enum):
    """
    A helper Enum for the mode we are in.
    """
    Manual = 0
    Exploration = 1
    ToStartLine = 2
    Race = 3
    Load = 4
    ToMarker = 5
    FollowPath = 6
    TurnToPoint = 7
    TurnAround = 8

class Main():
    def __init__(self) -> None:
        """
        
        """

        # load config
        self.config = load_config("config.yaml")

        # instantiate methods
        self.robot = RobotController(self.config)
        self.keypress_listener = KeypressListener()
        self.publisher = Publisher()
        self.PID = PIDController(self.config.pid_controller.P,self.config.pid_controller.I,self.config.pid_controller.D)
        #self.PID = PID(self.config.pid_controller.P,self.config.pid_controller.I,self.config.pid_controller.D)
        # set default values
        self.just_started = True
        self.DT = self.config.robot.delta_t # delta time in seconds
        self.expl = Explorer(50, 4, np.array((-1, 0)))
        self.count = 0
        self.speed = 0
        self.turn = 0
        self.new_speed = 0
        self.new_turn = 0
        self.triangulate = True
        self.degrees_to_turn = 0
        self.point_to_turn_to = np.array([0,0])
        self.turn_360_counter = 0
        self.manualMode = False
        self.is_running = True
        self.map = None
        self.middle = np.array([-1,0])
        self.mode = TaskPart.TurnAround
        self.goal_position = np.array([0,0])
        self.blocks_returned = 0
        # variable which saves the position of colored markers to drive to
        self.previous_color = 1 # 0 is red 1 is blue
        self.color_mask = np.empty(0)
        self.gate_mask = np.empty(0)
        self.block_marker_idx = []
        self.block_pos = np.empty(0)
        self.block_positions = []
        self.block_goal = np.empty(0)

        self.return_to_explo = False
        self.gate_pass = False
        self.doing_task = False
        self.do_triangulation = True
        self.came_from_triangulation = False

        self.path = []
        self.run_loop()

    def run_loop(self):
        """
        this loop wraps the methods that use the __enter__ and __close__ functions:
            self.keypress_listener, self.publisher, self.robot
        
        then it calls run()
        """
        print("starting...")

        # control vehicle movement and visualize it
        with self.keypress_listener, self.publisher, self.robot:
            print("starting EKF SLAM...")

            print("READY!")
            print("[green]MODE: Manual")
            

            count = 0
            """
            while self.is_running:
                time0 = time.time()
                elapsed_time = time.time() - time0
                #print(f"time: {elapsed_time}")
                self.run(count, time0, elapsed_time)
                if elapsed_time <= self.DT:
                    dt = self.DT - elapsed_time
                    time.sleep(dt) # moves while sleeping
                else:
                    print(f"[red]Warning! dt = {elapsed_time}")

                count += 1
                """
            while self.is_running:
                time0 = timer()
                self.run(count, time0)

                elapsed_time = timer() - time0
                if elapsed_time <= self.DT:
                    dt = self.DT - elapsed_time
                    time.sleep(dt) # moves while sleeping
            
                else:
                    #print(f"[red]Warning! dt = {elapsed_time}")
                    pass
                count += 1
    

            print("*** END PROGRAM ***")

    def check_point_inside_of_arena(self, point, edges, ids):

        filtered_edges = []
        
        for i in range(0, len(ids)):
            if 1 <= ids[i] <= 300:
                
                start_idx = i * 4
                filtered_edges.extend(edges[start_idx:start_idx + 4])
        
        polygon = np.array(filtered_edges, dtype=np.float32)
        inside = cv2.pointPolygonTest([polygon], (point[0], point[1]), False)
        return inside >= 0
    
    def can_drive_forward(self, data):
        if len(data.landmark_ids) == 0:
            return False

        marker_pos = data.landmark_positions[0]

        x = marker_pos[0]
        y = marker_pos[1]
        z = marker_pos[2] 
        distance  = np.sqrt(x*x + y*y)

        if distance < 0.1:
            return False
        return True
    
    def follow_marker(self, data):
        if len(data.landmark_ids) == 0:
            return 0,0

        marker_pos = data.landmark_positions[0]

        x = marker_pos[0]
        y = marker_pos[1]
        z = marker_pos[2] 
        distance  = np.sqrt(x*x + y*y)

        angle = data.landmark_alphas[0]
        """
        target_angle = 0

        speed = distance*100
        turn = self.PID.get_u(angle - target_angle, elapsed_time)

        """

        MAX_SPEED = 100
        MAX_TURN = 300
        DISTANCE_THRESHOLD = 0.05
        target_angle = 0
        ANGLE_THRESHOLD = 0.5
        #turn = angle * (MAX_TURN/np.pi)
        turn = self.PID.get_u(angle - target_angle)
        #turn = self.PID(-(angle - target_angle))
        turn = np.clip(turn, -MAX_TURN,MAX_TURN)
        speed = min(MAX_SPEED, 25 + distance*75)
        #print(distance, 'this is dt')
        
        if abs(angle) > ANGLE_THRESHOLD:
            speed = 0
        """
        else:
            if distance < 0.1:
                speed = 50
            else:
                speed = distance*100
            #speed = min(MAX_SPEED, distance*100)
            
            if distance < DISTANCE_THRESHOLD:
                speed = 0
        """
        #print(f"Distance: {distance:.2f}m, Angle: {np.degrees(angle):.1f}°")
        #print(f"Speed:{speed}, Turn: {turn}")
        #print(f"x: {x}, y: {y}, z: {z}")
        return speed, turn
    

    def update_gate_mask(self):
        # if the previous found color is red then we want to exclude the red gates from the next path planning and vice versa
        if self.previous_color == None:
            self.gate_mask = self.robot.slam.goal_ids // 100 == 4
        if self.previous_color == 0:
            self.gate_mask = self.robot.slam.goal_ids // 100 == 4
        elif self.previous_color == 1:
            self.gate_mask = self.robot.slam.goal_ids // 100 == 5

   
    def check_subsequent(self, marker_ids, marker_idx):

        marker_ids = marker_ids
        marker_idx = marker_idx

        sorted_indices = np.argsort(marker_ids)
        sorted_arr = marker_ids[sorted_indices]
    
        differences = np.diff(sorted_arr)
        subsequent_indices = np.where(differences == 1)[0]

        if subsequent_indices.size == 0:
            return False, [], []

        # Map back to the original indices of the elements and sort by original id value
        original_indices = np.array([
        tuple(sorted((sorted_indices[i], sorted_indices[i + 1]), key=lambda x: marker_ids[x]))
        for i in subsequent_indices
        ], dtype=np.int32)

        matching_markers = np.array([marker_ids[np.int32(key)] for key in original_indices])

        array_pos = np.array([marker_idx[np.int32(key)] for key in original_indices])

        return True, array_pos, matching_markers


    def check_for_match(self, marker_ids, marker_idx, num_times_seen):
        if self.previous_color == 1:
            mask = np.array([np.all(row // 100 == 4) for row in marker_ids])

        elif self.previous_color == 0:
            mask = np.array([np.all(row // 100 == 5) for row in marker_ids])

        else:
            mask = np.ones(len(marker_idx), dtype=bool)


        num_seen = np.array([[num_times_seen[np.int32(key1)], num_times_seen[np.int32(key2)]] for key1, key2 in marker_ids])
        print("This is mask and num_seen", mask, num_seen)
        i = 0
        for pair, idx, ids in zip(num_seen[mask], marker_idx[mask], marker_ids[mask]):
            if not np.any(pair <= 2):
                print(f'This is idx, ids for match {idx}, {ids}')
                return True, idx, ids, i
            i += 1

        return False, None, None, None


    # function to caluclate the position of a block if there is a match
    def estimate_block(self, marker_pos, distance = 0.1):
        block_positions = np.ones((len(marker_pos), 2))

        for i, pair in enumerate(marker_pos):
            #print(f'This is pair in block_estimation : {pair}')
            x1, y1 = pair[0]
            x2, y2 = pair[1]

            dx = x2 - x1
            dy = y2 - y1

            lenght = np.sqrt(dx**2 + dy**2)

            unit_vector = np.array([dx/lenght, dy/lenght])
            move_vector = unit_vector * distance
            block_positions[i] = np.array([x2 + move_vector[0], y2 + move_vector[1]])

        return block_positions


    # function to calculate the point where to drive the block to
    def return_to_gate(self, gate_pos, distance=0.3):
        # make matrix of euqlidian distances and choose max
        diffs = gate_pos[:, np.newaxis, :] - gate_pos[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
        idx_1, idx_2 = np.unravel_index(np.argmax(distances), distances.shape)
        print(idx_1, idx_2)

        # get points and calculate their vec
        p1, p2 = gate_pos[idx_1], gate_pos[idx_2]

        # make sure that p1 is the point nearer to start
        if p1[1] > p2[1]:
            p1, p2 = p2, p1

        print("This is p1 and p2", p1, p2)
        goal_vec = p2 - p1
        dx, dy = goal_vec

        # calc normalizes orthogonal vec and point where to go
        lenght = np.sqrt(dx**2 + dy**2)
        unit_orthogonal_vec = np.array([-dy/lenght, dx/lenght])
        move_vec = unit_orthogonal_vec * distance

        goal_point = np.array(p1 + np.array([0.5*dx, 0.5*dy]) - move_vec * distance)
        
        return goal_point
 
 
    ################################################################################################# Pauls code
    def check_for_one_marker(self, marker_ids, marker_idx, num_times_seen):
        if len(marker_ids) == 0:
            print("Skipped triangulation")
            return False, None

        if self.previous_color == 0:
            mask = np.array([np.all(row // 100 == 5) for row in marker_ids])
        
        if self.previous_color == 1:
            mask = np.array([np.all(row // 100 == 4) for row in marker_ids])

        elif self.previous_color == None:
            mask = np.ones_like(len(marker_idx), dtype=bool)[None]

        marker_ids = marker_ids[mask]
        marker_idx = marker_idx[mask]


            # Check for any single markers that have been seen enough times
        num_seen = np.array([num_times_seen[np.int32(key)] for key in marker_ids])
        valid_markers = num_seen > 5

        if np.any(valid_markers):
                # Return first valid marker position
            marker_idx = marker_idx[valid_markers][0]
            #print(f"single color idx: {marker_idx}")

            return True, marker_idx
        #print("no marker detecteds")
        return False, None
    

    def looking_at_point(self,data,point):

        delta_pos = point - data.robot_position
            
        angle_to_point = np.arctan2(delta_pos[1], delta_pos[0])
        angle = difference_angle(angle_to_point, data.robot_theta)
        if np.abs(angle) < 0.2:
            return True
        return False
    

    def calculate_equilateral_point(self, A, B, mid_point):
        dx = B[0] - A[0]
        dy = B[1] - A[1]
        
        midpoint = np.array([(A[0] + B[0])/2, (A[1] + B[1])/2])
        length = np.sqrt(dx**2 + dy**2)
        
        
        displacement_left = np.array([dy, -dx]) * (np.sqrt(3)/2)
        displacement_normalized_left = displacement_left * (length / np.linalg.norm(displacement_left))
        C_left = midpoint + displacement_normalized_left * 0.75

        displacement_right = np.array([-dy, dx]) * (np.sqrt(3)/2)
        displacement_normalized_right = displacement_right * (length / np.linalg.norm(displacement_right))
        C_right = midpoint + displacement_normalized_right * 0.75
        #print(f"point calculated: {C_right}")
        if np.linalg.norm(C_left-mid_point) < np.linalg.norm(C_right-mid_point):
            return C_left
        return C_right
    ################################################################################################# Pauls code


    def run(self, count, time0):
        """
        Were we get the key press, and set the mode accordingly.
        We can use the robot recorder to playback a recording.
        """

        if self.blocks_returned == self.config.blocks.count:
            for i in range(20):
                self.robot.move(-100,0)
            for i in range(100):
                self.robot.move(0,100)

        if not self.robot.recorder.playback:
            # read webcam and get distance from aruco markers
            _, raw_img, cam_fps, img_created = self.robot.camera.read() # BGR color
            # print(f"cam_fps: {cam_fps}")
            speed = self.speed
            turn = self.turn
        else:
            cam_fps = 0
            raw_img, speed, turn = next(self.robot.recorder.get_step)

        if raw_img is None:
            print("[red]image is None!")
            return
        self.parse_keypress(img=raw_img)
        
        if self.mode == TaskPart.Race:
            draw_img = raw_img
            data = self.robot.run_ekf_slam(raw_img, fastmode = True)
        else:
            draw_img = raw_img.copy()
            data = self.robot.run_ekf_slam(raw_img, draw_img)
            self.expl.update(data.robot_position)

            if not self.doing_task:
        # check if there is a pair of subsequent ids
                
                subsequent, subsequent_idx, subsequent_ids = self.check_subsequent(self.robot.slam.colored_ids, self.robot.slam.colored_idx)

                if subsequent:
                    colored_positions = np.array([[data.landmark_estimated_positions[np.int32(pair[0])], data.landmark_estimated_positions[np.int32(pair[1])]] for pair in subsequent_idx])
                    #print("This is colored positions : ", colored_positions, colored_positions.shape)

                    self.block_positions = self.estimate_block(colored_positions)
                    #print(f'These are the block cords : {self.block_positions}')
                    match, match_idx, match_ids, block_idx = self.check_for_match(subsequent_ids, subsequent_idx, self.robot.slam.num_times_seen_landmark)

                    if match:
                        print(f'Match with ids : {match_ids}')

                        red_gate_mask = np.where(self.robot.slam.goal_ids // 100 == 4, True, False)
                        blue_gate_mask = ~red_gate_mask 

                        if (match_ids[0] // 100 == 4 and len(self.robot.slam.goal_idx[red_gate_mask]) > 1) or (match_ids[0] // 100 == 5 and len(self.robot.slam.goal_idx[~blue_gate_mask] > 1)):

                            self.path = []
                            self.block_marker_idx = match_idx
                            
                            self.previous_color = int(match_ids[0] // 100 - 4) 
                            self.block_pos = np.append(self.block_pos, self.block_positions[block_idx])
                            self.block_positions = self.block_positions[1:]

                            if self.previous_color == 0:
                                mask = red_gate_mask

                            else:
                                mask = blue_gate_mask
                    
                            gate_landmarks = self.robot.slam.mu[3:].reshape(-1, 2)[self.robot.slam.goal_idx[mask]]
                            self.block_goal = np.append(self.block_goal, self.return_to_gate(gate_landmarks))

                            delete_index = np.where(np.isin(self.robot.slam.colored_ids, [match_ids[0],match_ids[1]]))
                            self.robot.slam.colored_idx = np.delete(self.robot.slam.colored_idx, delete_index)
                            self.robot.slam.colored_ids = np.delete(self.robot.slam.colored_ids, delete_index)

                            print(f'Found block at : {self.block_pos}. Delivering it to : {self.block_goal}. Block has color : {self.previous_color}')
                            self.doing_task = True
                            self.return_to_explo = True
                            self.mode = TaskPart.Exploration
                    ################################################################################################# Pauls code
                    
                elif not subsequent and self.do_triangulation:

                    match_one_marker, match_one_marker_idx = self.check_for_one_marker(self.robot.slam.colored_ids, self.robot.slam.colored_idx, self.robot.slam.num_times_seen_landmark)
                    if match_one_marker:
                        print("match single marker")
                        #colored_index = np.where(data.colored_markers_idx == match_one_marker_idx)[0][0]
                        point = data.landmark_estimated_positions[match_one_marker_idx]

                        if not self.looking_at_point(data,point):
                            self.mode = TaskPart.TurnToPoint
                            self.point_to_turn_to = point
                            self.triangulate = False
                        else:
                            print(f"Single colored marker is at: {point}")
                            self.triangulate = True
                            self.doing_task = True
                            self.came_from_triangulation = True
                            self.point_to_turn_to = point
                            self.mode = TaskPart.TurnToPoint
                            print("[green]MODE: TurnToPoint")

                        ################################################################################################# Pauls code

        if len(self.block_marker_idx) > 0 and len(self.path) > 0:
            block_pos_check = self.estimate_block(data.landmark_estimated_positions[self.block_marker_idx[None]])
            difference_block = np.linalg.norm(block_pos_check - self.path[-1])
            if difference_block > 0.03:
                print("Block_pos deviated by a lot, calculating new path to new block_pos")
                self.block_pos = block_pos_check[0]
                self.mode = TaskPart.Exploration
                self.path = []


           
        self.parse_keypress()
        #print(f"position: {data.robot_position}")

        if self.mode == TaskPart.Manual:
            self.parse_keypress()
            self.robot.move(self.speed, self.turn)
################################################################################################# Pauls code
        
        if self.mode == TaskPart.TurnAround:
            self.doing_task = True
            angle = data.robot_theta
            if -0.2 < angle < -0.01:
                self.mode = TaskPart.Exploration
                self.doing_task = False

            self.robot.move(0, 10)

            '''
            if abs(self.degrees_to_turn) > 0.1:
                angle = self.degrees_to_turn
                turn = self.PID.get_u(angle)
                theta1 = data.robot_theta
                self.robot.move(0, turn)
                theta2 = data.robot_theta
                self.degrees_to_turn -= np.abs(theta2 - theta1)

            else:

                self.mode = TaskPart.FollowPath
                print("[green]MODE: FOLLOWPATH")
            '''

        if self.mode == TaskPart.TurnToPoint:
            print("now in turn to point")
            delta_pos = self.point_to_turn_to - data.robot_position
            
            angle_to_point = np.arctan2(delta_pos[1], delta_pos[0])
            angle = difference_angle(angle_to_point, data.robot_theta)
            turn = self.PID.get_u(angle)
            speed = 0
            self.robot.move(speed, turn)
            
            if np.abs(angle) < .3:
                print("looking at the marker")
                if self.triangulate:
                    
                    triangulated_point = self.calculate_equilateral_point(self.point_to_turn_to, data.robot_position,self.middle)
                    self.goal_position = triangulated_point
                    print(f"triangulated point: {self.goal_position}")
                    self.path = []
                    self.mode = TaskPart.FollowPath
                    print("[green]MODE: FollowPath")
                else:
                    print("should look now at both markers")
                    self.triangulate = True
                    self.doing_task = False
                    self.do_triangulation = False
                    #self.TaskPart = TaskPart.Exploration
                    #print("[green]MODE: Exploration")


################################################################################################# Pauls code
        if self.mode == TaskPart.Exploration:
            
            marker_positions = np.array(data.landmark_positions)

            # if we have a task to fulfill we swap from exploration directly into path mode
            
            if len(self.block_pos) > 0:
                print("Driving to block")
                self.goal_position = self.block_pos
                self.block_pos = np.empty(0)
                self.mode = TaskPart.FollowPath
                print("[green]MODE: FOLLOWPATH")
            elif len(self.block_goal > 0):
                print("Returning block to goal!")
                self.block_marker_idx = []
                self.goal_position = self.block_goal
                self.block_goal = np.empty(0)
                self.gate_pass = True
                self.mode = TaskPart.FollowPath
                print("[green]MODE: FOLLOWPATH")


            elif self.return_to_explo == True:
                print("Returning to exploration coordinates!")
                if np.linalg.norm(data.robot_theta) > 0.2:
                    self.robot.move(0, -self.PID.get_u(data.robot_theta))
                else:
                    self.robot.move(-20,0)

                if(data.robot_position[0] < 0):
                    self.return_to_explo = False
                    self.doing_task = False
                    
            else:
                self.gate_pass = False
                ids = self.robot.slam.ids[self.robot.slam.ids != -1]
                  

                border_ids = ids[ids <= 100]
                border_ids = np.concat((ids[ids <= 100], self.robot.slam.goal_ids)).astype(int)
                border_idxs = self.robot.slam.ids_index[border_ids]
                borders = self.robot.slam.mu[3:].reshape(-1, 2)[border_idxs]

                obstacle_ids = ids[(ids >= 300) & (ids < 400)]
                obstacle_idxs = self.robot.slam.ids_index[obstacle_ids]
                obstacles = self.robot.slam.mu[3:].reshape(-1, 2)[obstacle_idxs]

                # check if the whole map has been explored
                if end_check(borders, mid=self.middle):
                    print("[green]Task has been completed")
                    sys.exit()

                
                #print(borders, obstacles)
                self.expl.update(data.robot_position)
                self.goal_position = self.expl.get_point(data.robot_position, borders, obstacles, num_candidates=50, obstacle_radius=0.15,  weight_expl=2.0, weight_dist=0.5,  weight_avoid=4.0)
                #self.goal_position = self.expl.get_point(data.robot_position, borders, obstacles, num_points=500)
                print(self.goal_position)
                self.mode = TaskPart.FollowPath 
                print("[green]MODE: FOLLOWPATH")
            '''
            elif len(marker_positions) > 0:
                distances = np.array(np.linalg.norm(marker_positions[:,:2]-data.robot_position[None], axis=1))
                farest_marker_idx = np.argmax(distances)
                farest_marker_position = marker_positions[farest_marker_idx][:2]
                
                vector_to_middle = (self.middle - farest_marker_position)
                normalized_vector_to_middle = vector_to_middle/np.linalg.norm(vector_to_middle)
                
                self.goal_position = farest_marker_position + normalized_vector_to_middle*0.25
                print(farest_marker_position)
                print(self.goal_position)
                time.sleep(5)
                self.mode = TaskPart.FollowPath
            else:
                self.robot.move(0,50)'''
            

        if self.mode == TaskPart.ToStartLine:
            self.mode = TaskPart.FollowPath

        if self.mode == TaskPart.Race:
            pass

        if self.mode == TaskPart.Load:
            pass

        if self.mode == TaskPart.FollowPath:
            ids = self.robot.slam.ids[self.robot.slam.ids != -1]
            if self.gate_pass == False:
                border_ids = np.concat((ids[ids <= 100], self.robot.slam.goal_ids)).astype(int)

            else:
                border_ids = ids[ids <= 100].astype(int)
            #print(border_ids)
                
            # adding the block positions as obstacles for A*

            border_idxs = self.robot.slam.ids_index[border_ids] 
            borders = self.robot.slam.mu[3:].reshape(-1, 2)[border_idxs]

            obstacle_ids = ids[(ids >= 300) & (ids < 400)]
            obstacle_idxs = self.robot.slam.ids_index[obstacle_ids]
            obstacles = self.robot.slam.mu[3:].reshape(-1, 2)[obstacle_idxs]
            if len(self.block_positions) > 0:
                obstacles = np.vstack((obstacles, self.block_positions))

            if self.gate_pass == True and len(self.path) == 0:
                '''gate_ids = self.robot.slam.goal_idx[self.gate_mask]
                print(gate_ids)
                landmarks_without_gates = np.ones(data.landmark_estimated_positions.shape[0], dtype=bool)
                landmarks_without_gates[gate_ids] = False
                print(landmarks_without_gates)'''

                self.path = combine_path(data.robot_position, self.goal_position, borders, obstacles)
                #print('path:', self.path)
                

            elif len(self.path) == 0:
                # check if goal position is outside of the arena
                #if self.check_point_inside_of_arena(self.goal_position,edges, nodes,data.landmark_estimated_ids) < 0:
                   # self.goal_position = (self.goal_position + self.middle) / 2.0
                self.path = combine_path(data.robot_position, self.goal_position, borders, obstacles)
                #print(self.path)

            #visable_borders = borders[np.isin(border_ids, data.landmark_ids)]
            #visalbe_obstacles = obstacles[np.isin(obstacle_ids, data.landmark_ids)]

            path_unblocked, end_inside = check_path(np.vstack((self.robot.slam.mu[:2], self.path)), self.middle, borders, obstacles, obstacle_radius=0.10)
            if not path_unblocked:
                if end_inside:
                    self.path = combine_path(data.robot_position, self.goal_position, borders, obstacles) 
                    # print(self.path)
                else:
                    self.path = []

            if end_inside:
                point = self.path[0]
                delta_pos = point - data.robot_position
                angle_to_point = np.arctan2(delta_pos[1], delta_pos[0])
                angle = difference_angle(angle_to_point, data.robot_theta)
                turn = self.PID.get_u(angle)
                speed = 15
                if np.abs(angle) > 0.06 * np.pi: # not sure which angle is appropriate here
                    speed = 0

                self.robot.move(speed, turn)

                if np.linalg.norm(delta_pos) < 0.02:
                    self.path = self.path[1:]
                    if len(self.path) == 0:
                        print("now at end path position")
                        ################################################################################################# Pauls code
                        
                        if self.came_from_triangulation == True:
                            print("came to triangulated point")
                            self.mode = TaskPart.TurnToPoint
                            print("[green]MODE: TurnToPoint")
                            self.came_from_triangulation = False
                            
                        else:
                            
                        #################################################################################################
                            self.mode = TaskPart.Exploration
                            print("[green]MODE: Exploration")
                            


            #self.drive_to_point(self.path[0], data.robot_position)


        # create a message for the viewer
        msg = Message(
            id = count,
            timestamp = time0,
            start = True,

            landmark_ids = data.landmark_ids,
            landmark_rs = data.landmark_rs,
            landmark_alphas = data.landmark_alphas,
            landmark_positions = data.landmark_positions,
            #goal_position=np.array(self.goal_position),
            
            landmark_estimated_ids = data.landmark_estimated_ids,
            landmark_estimated_positions = data.landmark_estimated_positions,
            landmark_estimated_stdevs = data.landmark_estimated_stdevs,

            robot_position = data.robot_position,
            robot_theta = data.robot_theta,
            robot_stdev = data.robot_stdev,
            #point_to_plot = data.goal_pos,
            text = f"cam fps: {cam_fps}",
            goal_position = self.goal_position
        )
        msg_str = jsonpickle.encode(msg)
        # send message to the viewer
        self.publisher.publish_img(msg_str, draw_img)

    def save_state(self, data):
        with open("SLAM.pickle", 'wb') as pickle_file:
            pass
        
        pass

    def load_and_localize(self):
        with open("SLAM.pickle", 'rb') as f:
            pass

        pass

    def parse_keypress(self,img=None):
        char = self.keypress_listener.get_keypress()

        turn_step = 10
        speed_step = 5

        if char == "a":
            if self.turn >= 0:
                self.new_turn = self.turn + turn_step
            else:
                self.new_turn = 0
            self.new_turn = min(self.new_turn, 200)
        elif char == "d":
            if self.turn <= 0:
                self.new_turn = self.turn - turn_step
            else:
                self.new_turn = 0
            self.new_turn = max(self.new_turn, -200)
        elif char == "w":
            if self.speed >= 0:
                self.new_speed = self.speed + speed_step
            else:
                self.new_speed = 0
            self.new_speed = min(self.new_speed, 100)
        elif char == "s":
            if self.speed <= 0:
                self.new_speed = self.speed - speed_step
            else:
                self.new_speed = 0
            self.new_speed = max(self.new_speed, -100)
        
        elif char == "q":
            self.new_speed = 0
            self.new_turn = 0
            self.is_running = False
        elif char == "m":
            self.new_speed = 0
            self.new_turn = 0
            self.mode = TaskPart.Manual
            print("[green]MODE: Manual")
        elif char == "r":
            self.mode = TaskPart.Race
            print("[green]MODE: Race")
        elif char == 'c':
            if img is None or not np.any(img):
                print("Image is empty or None, not saving.")
                return

            path = f"/home/pi/robotics-project-2024/robot-code/images/{self.count}_image.jpg"
            self.count += 1
            cv2.imwrite(path,img)
        elif char == "l":
            self.mode = TaskPart.Load
            print("[green]MODE: Load map")
        elif char == "k":
            self.mode = TaskPart.TurnAround
            print("[green]MODE: TurnAround")
            self.doing_task = True
        elif char == "p":
            self.mode = TaskPart.ToStartLine
            print("[green]MODE: To start line")
        elif char == "e":
            self.mode = TaskPart.Exploration
            print("[green]MODE: Exploration")
        elif char == "t":
            self.mode = TaskPart.FollowPath
            print("[green]MODE: To point")

        if self.speed != self.new_speed or self.turn != self.new_turn:
            self.speed = self.new_speed
            self.turn = self.new_turn
            print("speed:", self.speed, "turn:", self.turn)


if __name__ == '__main__':

    main = Main()

