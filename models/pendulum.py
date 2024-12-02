import numpy as np

class InvertedPendulum:
    def __init__(self, mass=1.0, length=1.0, gravity=9.81, time_step=0.05):
        """
        Initialize the inverted pendulum system.

        Parameters:
        - mass (float): Mass of the pendulum bob (kg).
        - length (float): Length of the pendulum rod (m).
        - gravity (float): Acceleration due to gravity (m/s^2).
        - time_step (float): Time step size (s).
        """
        self.mass = mass
        self.length = length
        self.gravity = gravity
        self.time_step = time_step

        # State variables: [angular position, angular velocity]
        self.angle = 0.0
        self.angular_velocity = 0.0

    def set_state(self, angle, angular_velocity):
        """
        Set the state of the pendulum.

        Parameters:
        - angle (float): Angular position of the pendulum (rad).
        - angular_velocity (float): Angular velocity of the pendulum (rad/s).
        """
        self.angle = angle
        self.angular_velocity = angular_velocity

        return self

    def get_state(self):
        """
        Retrieve the current state of the pendulum.

        Returns:
        - state (tuple): The current state [angle, angular velocity].
        """
        return self.angle, self.angular_velocity
    
    def update_dynamics(self, torque):
        """
        Update the state of the pendulum given control inputs.

        Parameters:
        - torque[0] (float): First control torque applied to the pendulum (N·m).
        - torque[1] (float): Second control torque applied to the pendulum (N·m).

        Returns:
        - state (tuple): The updated state [angle, angular velocity].
        """

        # Dynamics equations

        for u in torque:
            new_angle = self.angle + self.time_step * self.angular_velocity
            new_angular_velocity = self.angular_velocity + self.time_step * (- (self.gravity / self.length) * np.sin(-self.angle) + (1 / (self.mass * self.length**2)) * u)

            # Normalize angle to [-π, π]
            new_angle = (new_angle + np.pi) % (2 * np.pi) - np.pi

            self.angle = new_angle
            self.angular_velocity = new_angular_velocity
           
        return self.get_state()
    
    def max_jacobian(self, state, torque):
        """
        Compute the elements-wise absolute maximum of Jacobian matrix of the system dynamics with respect to the state variables.

        Parameters:
        - torque (list): List of control torques applied to the pendulum (N·m).

        Returns:
        - jacobian_matrix (np.ndarray): The the elements-wise absolute maximum of Jacobian matrix of the system dynamics.
        """
        # Partial derivatives of the dynamics equations with respect to angle and angular velocity
        d_angle_d_angle = 1
        d_angle_d_angular_velocity = self.time_step
        d_angular_velocity_d_angle = -self.time_step * (self.gravity / self.length) * np.cos(self.angle)
        d_angular_velocity_d_angular_velocity = 1

        # Construct the elements-wise absolute maximum Jacobian matrix
        d_angular_velocity_d_angle = self.time_step * (self.gravity / self.length)
        max_jacobian_matrix = np.array([
            [d_angle_d_angle, d_angle_d_angular_velocity],
            [d_angular_velocity_d_angle, d_angular_velocity_d_angular_velocity]
        ])

        return np.linalg.matrix_power(max_jacobian_matrix, 2)
