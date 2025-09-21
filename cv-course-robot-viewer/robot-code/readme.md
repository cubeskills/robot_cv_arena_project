# SLAM Robot

Designed to be used with [robot-viewer](https://gitlab.gwdg.de/cns-group/robot-viewer).

# Notes

```
camera.py
    runs the camera in a separate thread
    uses a lifo queue to always give the latest picture
        this is a bit weird, but easy to make thread safe.
    Important functions:
        read -> gets the camera image

EKFSLAM.py
    does EKFSLAM on the landmark detections
    Important functions:
        predict, add_landmark, correction -> SLAM
        get_robot_pose, get_landmark_poses -> get 2D position estomations

keypress_listener.py
    Exactly that.
    Buffered via stdin.
    Is called in run loop of main and delivers the keypresses.
    Important functions:
        get_keypress

opencv_utils.py
    putBText -> prints text on an image

recorder.py
    record the images and the driving instructions at that time

robot_controller.py
    Controlls the actual robot
    Important functions:
        move -> give command to move
        run_ekf_slam -> update SLAM
                            call as often as feasible
                            do before any decisions about how to move next

robot_dummy.py
    Stub for running without EV3

utils.py

vision.py
    Important functions:
        detections -> get the landmark positions
```
