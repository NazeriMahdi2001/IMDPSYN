#AI have been used to assist development of this code

import numpy as np

class VehicleDynamics:
    def __init__(self, x=0.0, y=0.0, theta=0.0, v=2.0, L=2.0, Ts=0.1):
        """
        Initialize the vehicle dynamics model.

        Args:
        x (float): Initial position in the x-direction.
        y (float): Initial position in the y-direction.
        theta (float): Initial orientation (heading) of the vehicle in radians.
        v (float): Initial velocity of the vehicle.
        L (float): Wheelbase of the vehicle (distance between front and rear axle).
        Ts (float): Time step for discrete-time dynamics.
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.L = L
        self.Ts = Ts

    def update_dynamics(self, delta):
        """
        Update the vehicle's state using discrete-time non-linear dynamics.

        Args:
        delta (float): Steering angle input in radians.

        Returns:
        (float, float, float): Updated state (x, y, theta).
        """
        # Compute the new state based on the previous state and inputs
        x_next = self.x + self.Ts * self.v * np.cos(self.theta)
        y_next = self.y + self.Ts * self.v * np.sin(self.theta)
        theta_next = self.theta + self.Ts * (self.v / self.L) * np.tan(delta)
        # v_next = self.v + self.Ts * a
        
        # Update the current state with new values
        self.x = x_next
        self.y = y_next

        while theta_next > np.pi:
            theta_next -= 2 * np.pi
        while theta_next < -np.pi:
            theta_next += 2 * np.pi
        self.theta = theta_next

        return self.x, self.y, self.theta

    def get_state(self):
        """
        Get the current state of the vehicle.

        Returns:
        (float, float, float): Current state (x, y, theta).
        """
        return self.x, self.y, self.theta
