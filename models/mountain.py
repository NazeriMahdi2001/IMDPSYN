import numpy as np

class MountainCar:
    def __init__(self):
        """
        Initialize the Mountain Car environment.
        """
        self.min_position = -1.2
        self.max_position = 0.6
        self.min_velocity = -0.07
        self.max_velocity = 0.07

        self.force = 0.001
        self.gravity = -0.0025

        self.n_steps = 8

        self.pos = 0
        self.vel = 0

    def set_state(self, pos, vel):
        """
        Set the state of the environment.

        Parameters:
            position (float): The position of the car.
            velocity (float): The velocity of the car.
        """
        self.pos = pos
        self.vel = vel

        return self

    def get_state(self):
        """
        Return the current state.

        Returns:
            state (ndarray): The current state [position, velocity].
        """
        return self.pos, self.vel
    
    def update_dynamics(self, action):
        """
        Update the environment state based on the action.

        Parameters:
            action (int): The action to take. Must be -1 (left), 0 (no acceleration), or +1 (right).

        Returns:
            state (ndarray): The next state [position, velocity].
            reward (float): The reward received.
            done (bool): Whether the episode has ended.
        """
        for a in action:
            new_velocity = self.vel + self.force * a + self.gravity * np.cos(3 * self.pos)
            new_velocity = np.clip(new_velocity, self.min_velocity, self.max_velocity)

            # Update position
            new_position = self.pos + self.vel
            new_position = np.clip(new_position, self.min_position, self.max_position)

            # Update state
            self.pos = new_position
            self.vel = new_velocity

        return self.get_state()

    def lipschitz(self):
        return np.linalg.matrix_power(np.array([[1, 1], [self.gravity, 1]]), self.n_steps)