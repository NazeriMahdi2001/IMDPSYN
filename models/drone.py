import numpy as np

class Drone:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def set_state(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

        return self

    def get_state(self):
        return self.x, self.y, self.z
    
    def update_dynamics(self, control):

        new_x = self.x + 10 * control[0] * np.cos(control[1])
        new_y = self.y + 10 * control[0] * np.sin(control[1])
        new_z = self.z + 10 * control[2]

        self.x = new_x
        self.y = new_y
        self.z = new_z
        
        return self.get_state()
    
    def max_jacobian(self, state, control):
        # Partial derivatives of the dynamics equations with respect to angle and angular velocity
        d_1_d_1 = 1
        d_1_d_2 = 0
        d_1_d_3 = 0

        d_2_d_1 = 0
        d_2_d_2 = 1
        d_2_d_3 = 0

        d_3_d_1 = 0
        d_3_d_2 = 0
        d_3_d_3 = 1

        max_jacobian_matrix = np.array([
            [d_1_d_1, d_1_d_2, d_1_d_3],
            [d_2_d_1, d_2_d_2, d_2_d_3],
            [d_3_d_1, d_3_d_2, d_3_d_3]
        ])

        return np.linalg.matrix_power(max_jacobian_matrix, 1)
