import numpy as np
from rich import print
from timeit import default_timer as timer
import copy
from utils.vision import Vision

def difference_angle(angle1, angle2):
    difference = (angle1 - angle2) % (2*np.pi)
    difference = np.where(difference > np.pi, difference - 2*np.pi, difference)
    difference = difference.item() # we want to convert 0-d array to float

    return difference

class EKFSLAM:
    def __init__(self,
        WHEEL_RADIUS,
        WIDTH,

        MOTOR_STD,
        DIST_STD,
        ANGLE_STD,

        init_state: np.ndarray = np.zeros(3),
        init_covariance: np.ndarray = np.zeros((3,3))
    ):
        self.WHEEL_RADIUS = WHEEL_RADIUS
        self.WIDTH = WIDTH

        self.mu = init_state.copy()
        self.Sigma = init_covariance.copy()
        self.ids = np.full((1000,), -1) # create an array instead of a list
        self.ids_index = np.full((2000,), -1) # lookup table to get index of id # assume, no ids larger than 2000
        self.num_ids = 0

        # variables to keep track of 'colored' markers
        self.colored_idx = np.empty(0, dtype=int)
        self.colored_ids = np.empty(0)
        self.red_mask = np.ones(0, dtype=bool)
        self.goal_idx = np.empty(0, dtype=int)
        self.goal_ids = np.empty(0)

        

        self.DIST_STD = DIST_STD
        self.ANGLE_STD = np.radians(ANGLE_STD)

        self.error_l = np.radians(MOTOR_STD) * WHEEL_RADIUS * np.sqrt(2)
        self.error_r = np.radians(MOTOR_STD) * WHEEL_RADIUS * np.sqrt(2)

        self.num_times_seen_landmark = {}

        return

    def predict(self, l, r):
        time1 = timer()

        x, y, theta, std = self.get_robot_pose()
         # l, r = u
        alpha = (r-l) / self.WIDTH

        if abs(r - l) >= np.radians(1) * self.WHEEL_RADIUS: # difference in left and right angle > 1°
            ### Your code here ###
            # should include G, A, B, C, D, x, y, theta
            
            R = l/alpha
            x = x + (R + self.WIDTH/2)*(np.sin(theta + alpha) - np.sin(theta))
            y = y + (R + self.WIDTH/2)*(-np.cos(theta + alpha) + np.cos(theta))
            theta = (theta + alpha) % (2 * np.pi)
            if theta > np.pi:
                theta = theta-2*np.pi
            
            G = np.array([
                [1, 0, (l/alpha + self.WIDTH/2)* (np.cos(theta + alpha) - np.cos(theta))],
                [0, 1, (l/alpha + self.WIDTH/2)* (np.sin(theta + alpha) - np.sin(theta))],
                [0, 0, 1]
            ])
            A = (self.WIDTH*r)/((r-l)**2)*(np.sin(theta + (r-l)/self.WIDTH) - np.sin(theta)) - (r+l)/(2*(r-l))*np.cos(theta + (r-l)/self.WIDTH)
            B = (self.WIDTH*r)/((r-l)**2)*(-np.cos(theta + (r-l)/self.WIDTH) + np.cos(theta)) - (r+l)/(2*(r-l))*np.sin(theta + (r-l)/self.WIDTH)
            C = -(self.WIDTH*l)/((r-l)**2)*(np.sin(theta + (r-l)/self.WIDTH) - np.sin(theta)) + (r+l)/(2*(r-l))*np.cos(theta + (r-l)/self.WIDTH)
            D = -(self.WIDTH*l)/((r-l)**2)*(-np.cos(theta + (r-l)/self.WIDTH) + np.cos(theta)) + (r+l)/(2*(r-l))*np.sin(theta + (r-l)/self.WIDTH)

          
            ###
        else:
            ### Your code here ###
            # should include G, A, B, C, D, x, y
            
            x = x + l * np.cos(theta)
            y = y + l * np.sin(theta)

            G = np.array([
                [1, 0, -l * np.sin(theta)],
                [0, 1, l * np.cos(theta)],
                [0, 0, 1]
            ])
            A = 0.5 *(np.cos(theta) + l/self.WIDTH*np.sin(theta))
            B = 0.5 *(np.sin(theta) - l/self.WIDTH*np.cos(theta))
            C = 0.5 *(np.cos(theta) - l/self.WIDTH*np.sin(theta))
            D = 0.5 *(np.sin(theta) + l/self.WIDTH*np.cos(theta))


            ###
        V = np.array([
            [A, C],
            [B, D],
            [-1/self.WIDTH, 1/self.WIDTH]])

        N = self.num_ids
        if N > 0:
            G = np.block([[G, np.zeros((3,2*N))], [np.zeros((2*N,3)), np.identity(2*N)]])
            V = np.append(V, np.zeros((2*N,2)), axis=0)

        self.mu[:3] = x, y, theta
        diag = np.diag(np.array([np.power(self.error_l,2), np.power(self.error_r,2)]))
        self.Sigma = np.dot(np.dot(G, self.Sigma), G.T) + np.dot(np.dot(V, diag), V.T)

        elapsed_time = timer() - time1
        if elapsed_time > 0.1:
            print(f"[red]EKF_SLAM predict time: {elapsed_time}")

    def add_landmark(self, position: tuple, measurement: tuple, id: int):
        if id == 0 or id >= 600:
            return
        x, y, z = position
        #print(f"x: {x}, y: {y}, z: {z}")
        # add with variance of 100m
        x_var = float(100)
        y_var = float(100)
        

        self.mu = np.append(self.mu, [x,y])
        self.Sigma = np.block([
            [self.Sigma, np.zeros((self.Sigma.shape[0], 2))],
            [np.zeros((2, self.Sigma.shape[1])), np.diag(np.array([x_var, y_var]))]
        ])
        #print(self.mu)
        # add the id to the array
        self.ids[self.num_ids] = id
        self.ids_index[id] = self.num_ids

        # add check to see if marker is a colored marker
        if 400 <= id <= 600 and id % 100 > 20:
            self.colored_idx = np.append(self.colored_idx, self.num_ids)
            self.colored_ids = np.append(self.colored_ids, id)
            if id // 100 == 4:
                self.red_mask = np.append(self.red_mask, True)
            else:
                self.red_mask = np.append(self.red_mask, False)


        elif 400 <= id <= 600 and id % 100 < 20:
            self.goal_idx = np.append(self.goal_idx, self.num_ids)
            self.goal_ids = np.append(self.goal_ids, id)

        #print('Added landmark number :', self.num_ids, '; with id :', id )
        #print('Coordinates : ', x,';   ', y)
        self.num_ids += 1

        self.num_times_seen_landmark[id] = 1

    
    def correction(self, landmark_position_measured: tuple, id: int, count=True):
        time1 = timer()

        r_meas, beta_meas = landmark_position_measured
        
        N = self.num_ids
        #print(f"N: {N}")
        i = self.ids_index[id]
        #print(f"i: {i}")
        x_lm, y_lm = self.mu[3+2*i : 3+2*(i+1)]
        # x_bot, y_bot, theta_bot, stdev_bot = self.get_robot_pose()
        x_bot,y_bot,theta_bot = self.mu[:3].copy()

        dx = x_lm - x_bot
        dy = y_lm - y_bot

        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        beta = np.arctan2(dy,dx) - theta_bot
        
        H_small = np.zeros((2,5))
        H_small[0,:3] = np.array([-dx / r, -dy / r, 0])
        H_small[1,:3] = np.array([dy / r**2, -dx / r**2, -1])

        H_small[0,3 : 5] = np.array([dx / r, dy / r])
        H_small[1,3 : 5] = np.array([-dy / r**2, dx / r**2])

        H = np.zeros((2,3+2*N))
        H[0,:3] = np.array([-dx / r, -dy / r, 0])
        H[1,:3] = np.array([dy / r2, -dx / r2, -1])

        H[0,3+2*i : 3+2*(i+1)] = np.array([dx / r, dy / r])
        H[1,3+2*i : 3+2*(i+1)] = np.array([-dy / r2, dx / r2])
        
        sigma_small = np.zeros((5, 5))

        sigma_small[:3, :3] = self.Sigma[:3, :3]
        sigma_small[3:5, 3:5] = self.Sigma[3+2*i : 3+2*(i+1), 3+2*i : 3+2*(i+1)]

        # cross-variances
        sigma_small[3:5, 0:3] = self.Sigma[3+2*i : 3+2*(i+1), 0:3]
        sigma_small[0:3, 3:5] = self.Sigma[0:3, 3+2*i : 3+2*(i+1)]

        
       
        
        diff_in_angle = difference_angle(beta_meas, beta)
        diff_in_r = r_meas - r
        Q = np.diag([self.DIST_STD, self.ANGLE_STD])
        
        Z = np.linalg.inv(H_small @ sigma_small @ H_small.T + Q)

        K = self.Sigma @ H.T @ Z
        
        err = np.array([diff_in_r, diff_in_angle])
        correction = K @ err

        self.mu += correction
        self.Sigma = (np.identity(3+2*N) - K @ H) @ self.Sigma
        

        elapsed_time = timer() - time1
        if elapsed_time > 0.1:
            print(f"[red]EKF_SLAM correction time: {elapsed_time}")

        # count how many times we have found it
        if count:
            self.num_times_seen_landmark[id] += 1
    

    def get_robot_pose(self):
        x,y,theta = self.mu[:3]
        sigma = self.Sigma[:2, :2]
        error = self.get_error_ellipse(sigma)
        #print(f"theta: {theta}")

        return x, y, theta, error

    def get_landmark_poses(self, at_least_seen_num=0):
        positions = self.mu[3:].copy()
        positions = positions.reshape((len(positions) // 2, 2))

        errors = []
        for i in np.arange(self.num_ids):
            j = 3 + 2 * i
            sigma_i = self.Sigma[j:j+2, j:j+2]

            errors.append(self.get_error_ellipse(sigma_i))

        if at_least_seen_num == 0 or at_least_seen_num == 1:
            return positions, np.array(errors), np.array(self.ids[:self.num_ids])
        else:
            # only return landmarks that have been seen at least twice
            mask = np.zeros(self.num_ids, dtype=bool)
            for i, id in enumerate(self.ids):
                if id == -1:
                    break
                if self.num_times_seen_landmark[id] >= at_least_seen_num:
                    mask[i] = 1

            return positions[mask], np.array(errors)[mask], np.array(self.ids[:self.num_ids])[mask]


    def get_landmark_pose(self, id):
        i = self.ids_index[id]
        positions = self.mu[3:]

        j = 3 + 2 * i
        sigma_i = self.Sigma[j:j+2, j:j+2]

        return positions[i*2:i*2 + 2], self.get_error_ellipse(sigma_i)



    def get_error_ellipse(self, covariance):
        all_zeros = not np.any(covariance)
        if all_zeros:
            return 0, 0, 0
        else:
            eigen_vals, eigen_vecs = np.linalg.eig(covariance)

            # angle of first (largest) eigenvector
            angle = np.arctan2(eigen_vecs[1, 0], eigen_vecs[0, 0])

        return np.sqrt(eigen_vals[0]), np.sqrt(eigen_vals[1]), angle


    def get_landmark_ids(self):
        return np.array(self.ids[:self.num_ids])

    def remove_blocks(self):
        for landmark_id in self.ids:
            if landmark_id > -1 and landmark_id >= 1000:
                self.remove_by_id(landmark_id)
                

    def remove_by_id(self, landmark_id):

        print("[red]removing landmark ", landmark_id)

        # get index
        i = self.ids_index[landmark_id]

        self.ids = np.delete(self.ids, i) # same as pop

        # recreate index array
        self.ids_index[landmark_id] = -1
        for an_idx, an_id in enumerate(self.ids):
            if an_id == -1:
                break

            self.ids_index[an_id] = an_idx

        del self.num_times_seen_landmark[landmark_id]

        self.num_ids -= 1

        # remove the block from mu
        self.mu = np.hstack((self.mu[:3+2*i], self.mu[3+2*(i+1):]))

        # we remove the row and column corresponding to the 2x2 Sigma block
        self.Sigma = np.block([
            [self.Sigma[:3+2*i, :3+2*i], self.Sigma[:3+2*i, 3+2*(i+1):]],
            [self.Sigma[3+2*(i+1):, :3+2*i], self.Sigma[3+2*(i+1):, 3+2*(i+1):]]
        ])
