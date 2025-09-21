# Computer Vision and Robotics Project
## University of Göttingen - CV&R Course

This repository contains the implementation of an autonomous robot navigation system developed as part of the Computer Vision and Robotics course at the University of Göttingen. The project implements SLAM (Simultaneous Localization and Mapping) using ArUco marker detection for autonomous robot exploration and task completion.

## Project Overview

The project consists of developing a complete autonomous robot system that can:
- **Navigate autonomously** in an unknown environment
- **Detect and localize ArUco markers** for environmental mapping
- **Build real-time maps** using EKF-SLAM (Extended Kalman Filter SLAM)
- **Plan optimal paths** and execute complex navigation tasks
- **Perform object manipulation** tasks using colored block detection

## System Architecture

### Core Components

#### 1. **Robot Viewer** (`viewer.py`)
- Real-time visualization system with multiple canvases
- Displays robot trajectory, sensor data, camera feed, and world map
- Supports both live robot connection and local simulation modes
- Built with Tkinter GUI for comprehensive monitoring

#### 2. **Robot Controller** (`robot-code/main.py`)
- Main control loop implementing different operational modes:
  - Manual control
  - Autonomous exploration
  - Task-specific navigation (ToMarker, FollowPath, etc.)
- Integrates EKF-SLAM with PID control for precise movement
- Handles block detection and manipulation tasks

#### 3. **EKF-SLAM Implementation** (`utils/EKFSLAM.py`)
- Extended Kalman Filter for simultaneous localization and mapping
- Real-time pose estimation and landmark mapping
- Handles data association and loop closure detection

#### 4. **Computer Vision Pipeline** (`utils/vision.py`)
- ArUco marker detection and pose estimation
- Camera calibration and image undistortion
- Real-time image processing for navigation


## 🛠️ Technical Implementation

### Task Progression (Jupyter Notebooks)

1. **ArUco Detection** (`1_aruco_detection.ipynb`)
   - Marker detection algorithms
   - Camera calibration pipeline
   - Pose estimation validation

2. **Coordinate Transformation** (`2_coordinate_transform.ipynb`)
   - World coordinate system setup
   - Camera-to-world transformations
   - Calibration verification

3. **Robot Control** (`3_driving.ipynb`)
   - Basic robot movement
   - Motor control testing
   - Safety protocols

4. **PID Control** (`4_PID.ipynb`)
   - PID controller tuning
   - Performance optimization
   - Stability analysis

5. **SLAM Implementation** (`5_SLAM.ipynb`)
   - EKF-SLAM algorithm
   - Real-time mapping
   - Localization accuracy

6. **Autonomous Navigation** (`6_drive_to_real_world_coords.ipynb`)
   - Point-to-point navigation
   - Path execution
   - Error correction

7. **Map Discretization** (`7_discretise_real_world_map.ipynb`)
   - Grid-based map representation
   - Obstacle mapping
   - Path planning preparation

8. **Complete System Integration** (`8_apply_planning.ipynb`)
   - Full autonomous behavior
   - Task scheduling
   - Performance evaluation

## 🚀 Getting Started

### Prerequisites
```bash
pip install opencv-python opencv-contrib-python
pip install numpy matplotlib PyYAML
pip install tkinter pillow jsonpickle
pip install zmq rich numba
```

### Running the System

#### Live Robot Operation
```bash
# On the robot
cd robot-code
python main.py

# On the control station
python viewer.py
```

#### Simulation Mode
```bash
# Run SLAM simulation
python viewer.py --local

# Then execute the SLAM notebook for data playback
```

### Configuration
Edit `robot-code/config.yaml` to adjust:
- Robot physical parameters (wheel radius, width)
- Camera mounting position and orientation
- SLAM algorithm parameters
- PID controller gains
- Task-specific settings

## Results & Performance

The system successfully demonstrates:
- **Accurate SLAM**: Sub-centimeter localization accuracy
- **Robust Navigation**: Reliable obstacle avoidance and path following
- **Task Completion**: Autonomous block manipulation and goal reaching
- **Real-time Performance**: 10Hz control loop with live visualization

## Project Achievements

This implementation showcases advanced robotics concepts including:
- Probabilistic state estimation (EKF)
- Computer vision integration
- Real-time control systems
- Path planning algorithms
- Human-robot interface design

## Project Structure

```
cv-course-robot-viewer/
├── viewer.py              # Main visualization system
├── subscriber.py          # Data communication
├── drawables.py          # Visualization components
├── message.py            # Communication protocol
└── robot-code/
    ├── main.py           # Robot control system
    ├── config.yaml       # System configuration
    ├── publisher.py      # Data broadcasting
    ├── continuous.py     # Path planning algorithms
    ├── utils/            # Core algorithms
    │   ├── EKFSLAM.py   # SLAM implementation
    │   ├── vision.py    # Computer vision
    │   ├── robot_controller.py  # Hardware interface
    │   └── PID.py       # Control algorithms
    ├── notebooks/        # Development and testing
    └── images/          # Test data and results
```

## 👥 Course Information

**University**: Georg-August-Universität Göttingen  
**Course**: Computer Vision and Robotics  
**Implementation**: Framework specialization for autonomous navigation tasks

This project demonstrates the practical application of theoretical computer vision and robotics concepts in a real-world autonomous system.