from __future__ import annotations
from types import SimpleNamespace
import ev3_dc as ev3
import numpy as np
from rich import print


from utils.camera import Camera
from utils.vision import Vision
from utils.EKFSLAM import EKFSLAM
from utils.recorder import Recorder
from utils.robot_dummy import DummyVehicle

from timeit import default_timer as timer


class RobotController:
    def __init__(self, config) -> None:

        self.config = config
        self.dt = config.robot.delta_t

        self.__ev3_obj__ = None
        self.vehicle = None

        self.camera = None
        self.vision = None

        self.recorder = Recorder(self.dt)

        self.slam = EKFSLAM(
            config.robot.wheel_radius,
            config.ekf_slam.robot_width,
            MOTOR_STD=config.ekf_slam.motor_std,
            DIST_STD=config.ekf_slam.dist_std,
            ANGLE_STD=config.ekf_slam.angle_std
        )

        self.old_l, self.old_r = 0, 0

        self.detected_ids = set()


    def __enter__(self) -> RobotController:

        self.camera = Camera(self.config.camera.exposure_time,
                             self.config.camera.gain)
        self.vision = Vision(self.camera.CAMERA_MATRIX, self.camera.DIST_COEFFS,
                             self.config.camera)

        try:
            self.__ev3_obj__ = ev3.EV3(protocol=ev3.USB, sync_mode="STD")
        except Exception as e:
            
            print("error:", e)

        if self.__ev3_obj__:
            self.vehicle = ev3.TwoWheelVehicle(
                self.config.robot.wheel_radius, # radius wheel
                self.config.robot.width, # middle-to-middle tread measured
                speed = 10,
                ev3_obj=self.__ev3_obj__
            )

            print("[green]***CONNECTED TO REAL VEHICLE***[/green]")
        else:
            print("[red]***USING Dummy VEHICLE***[/red]")
            self.vehicle = DummyVehicle()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.recorder.is_recording:
            self.recorder.save_recording()

        if self.vehicle:
            self.vehicle.stop(brake=False)
            self.vehicle.__exit__(exc_type, exc_val, exc_tb)

        if self.__ev3_obj__:
            self.__ev3_obj__.__exit__(exc_type, exc_val, exc_tb)

        if self.camera:
            self.camera.close()

    def wheel_rotation_from_distance(self, d):
        """
        @brief: Calculates the rotation of a robot wheel from the distance that wheel has traveled and the wheel diameter
        @param d: Distance traveled
        @return: Angle by which the wheel has turned while traveling distance d
        """
        wheel_circumference = 2*self.slam.WHEEL_RADIUS*np.pi
        revolutions = d / wheel_circumference

        return 2*np.pi * revolutions
    
    def move(self, speed, turn, img=None, dt=None):
        x, y, theta, _ = self.slam.get_robot_pose()
        if not dt:
            dt = self.dt

        v_max = 1
        w_max = 4*np.pi
        v = np.clip(v_max * speed / 100, -1 * v_max, v_max)
        w = np.clip(w_max * turn / 100, -1 * w_max, w_max)
        if speed == 0 and turn == 0:
            self.vehicle.stop()
            return
        if w == 0:
            vl = v
            vr = v
        else:
            vl = w*(v/w - self.slam.WIDTH/2)
            vr = w*(v/w + self.slam.WIDTH/2)

        vr = int(np.copysign(np.ceil(np.abs(100 * vr)), vr))
        vl = int(np.copysign(np.ceil(np.abs(100 * vl)), vl))
        # normalise to avoid overflow
        if np.abs(vr) > 100 or np.abs(vl) > 100:
            vr, vl = np.round(100 * np.array([vr, vl]) / np.max(np.abs([vr, vl])))
        ops = b''.join((
            ev3.opOutput_Step_Speed,
            ev3.LCX(0),  # LAYER
            ev3.LCX(self.vehicle.port_left),  # NOS
            ev3.LCX(int(vl)),
            ev3.LCX(50),  # STEP1
            ev3.LCX(50),  # STEP2
            ev3.LCX(50),  # STEP3
            ev3.LCX(0),  # BRAKE
            
            ev3.opOutput_Step_Speed,
            ev3.LCX(0),  # LAYER
            ev3.LCX(self.vehicle.port_right),  # NOS
            ev3.LCX(int(vr)),
            ev3.LCX(50),  # STEP1
            ev3.LCX(50),  # STEP2
            ev3.LCX(50),  # STEP3
            ev3.LCX(0),  # BRAKE
            
            ev3.opOutput_Start,
            ev3.LCX(0),  # LAYER
            ev3.LCX(self.vehicle.port_left + self.vehicle.port_right),  # NOS
        ))

        self.vehicle.send_direct_cmd(
                self.vehicle._ops_pos() + ops,
                global_mem=8
            )
        


        self.recorder.save_step(img, speed, turn, x, y, theta, self.old_l, self.old_r)

    def move_old(self, speed, turn, img=None):
        self.vehicle.move(speed, turn)
        self.recorder.save_step(img, speed, turn)

    def get_motor_movement(self) -> tuple:
        new_l,new_r = self.vehicle.motor_pos
    
        dl, dr = new_l-self.old_l, new_r-self.old_r
        circumfrance = self.config.robot.wheel_radius*2*np.pi
        dl,dr = dl/360*circumfrance, dr/360*circumfrance
        self.old_l,self.old_r = new_l,new_r
        # print(f"dl: {dl}, dr: {dr}")
        return (dl,dr)

    
    def run_ekf_slam(self, img, draw_img=None, fastmode=False):
        # movements is what is refered to as u = (l, r) in the document
        movements = self.get_motor_movement()
        
        if movements[0] != 0.0 or movements[1] != 0:
            self.slam.predict(*movements)
        
        #################################################
        if img is not None:# might need to be deleted
        ###############################################
            ids, landmark_rs, landmark_alphas, landmark_positions = self.vision.detections(img, draw_img, self.slam.get_robot_pose())

        for i, id in enumerate(ids):
            ################################# Coord transform to world
            robot_x, robot_y, robot_theta, _ = self.slam.get_robot_pose()
            landmark_positions[i][0] = robot_x + np.cos(robot_theta + landmark_alphas[i]) * landmark_rs[i]
            landmark_positions[i][1] = robot_y + np.sin(robot_theta + landmark_alphas[i]) * landmark_rs[i]

            if id not in self.slam.get_landmark_ids():
                self.slam.add_landmark(landmark_positions[i], (landmark_rs[i], landmark_alphas[i]), id)
            else:
                # correct each detected landmark that is already added
                self.slam.correction((landmark_rs[i], landmark_alphas[i]), id)

        robot_x, robot_y, robot_theta, robot_stdev = self.slam.get_robot_pose()
        landmark_estimated_ids = self.slam.get_landmark_ids()
        colored_markers_idx = self.slam.colored_idx
        colored_markers_ids = self.slam.colored_ids
        gates_idx = self.slam.goal_idx
        gates_ids = self.slam.goal_ids
        landmark_estimated_positions, landmark_estimated_stdevs, ids2 = self.slam.get_landmark_poses()
        #print(landmark_estimated_positions)
        

        data = SimpleNamespace()
        data.landmark_ids = ids
        data.landmark_rs = landmark_rs
        data.landmark_alphas = landmark_alphas
        data.landmark_positions = landmark_positions
        data.landmark_estimated_ids = landmark_estimated_ids
        data.landmark_estimated_positions = landmark_estimated_positions
        data.landmark_estimated_stdevs = landmark_estimated_stdevs
        #print(f"alpha: {landmark_alphas[0]}, radius: {landmark_rs[0]}, positions: {landmark_positions}")
        data.colored_markers_idx = colored_markers_idx
        data.colored_markers_ids = colored_markers_ids
        data.gates_idx = gates_idx
        data.gates_ids = gates_ids
        data.robot_position = np.array([robot_x, robot_y])
        data.robot_theta = robot_theta
        data.robot_stdev = robot_stdev
        #data.goal_pos = [0,0]

        return data
