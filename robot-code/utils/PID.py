import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

class PIDController:
    def __init__(self,Kp,Ki,Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_error = 0
        self.last_error = 0

    def get_u(self,error,dt):
        self.integral_error += error*dt 
        self.deriv_error = (error-self.last_error)/dt
        self.last_error = error
        
        return self.Kp * error + self.Ki * self.integral_error + self.Kd * self.deriv_error






