import numpy as np

class DubinsCar:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def set_state(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

        return self

    def get_state(self):
        return self.x, self.y, self.theta
    
    def update_dynamics(self, control):

        new_x = self.x + 10 * control[0] * np.cos(self.theta)
        new_y = self.y + 10 * control[0] * np.sin(self.theta)
        new_theta = self.theta + control[1]

        self.x = new_x
        self.y = new_y
        
        new_x = self.x + 10 * control[2] * np.cos(self.theta)
        new_y = self.y + 10 * control[2] * np.sin(self.theta)
        new_theta = self.theta + control[3]

        self.x = new_x
        self.y = new_y
        
        return self.get_state()
    
    def max_jacobian(self, state, control):
        # Partial derivatives of the dynamics equations with respect to angle and angular velocity
        d_1_d_1 = 1
        d_1_d_2 = 0
        d_1_d_3 = 10 * control[0]

        d_2_d_1 = 0
        d_2_d_2 = 1
        d_2_d_3 = 10 * control[0]

        d_3_d_1 = 0
        d_3_d_2 = 0
        d_3_d_3 = 1

        max_jacobian_matrix1 = np.array([
            [d_1_d_1, d_1_d_2, d_1_d_3],
            [d_2_d_1, d_2_d_2, d_2_d_3],
            [d_3_d_1, d_3_d_2, d_3_d_3]
        ])

        d_1_d_1 = 1
        d_1_d_2 = 0
        d_1_d_3 = 10 * control[2]

        d_2_d_1 = 0
        d_2_d_2 = 1
        d_2_d_3 = 10 * control[2]

        d_3_d_1 = 0
        d_3_d_2 = 0
        d_3_d_3 = 1

        max_jacobian_matrix2 = np.array([
            [d_1_d_1, d_1_d_2, d_1_d_3],
            [d_2_d_1, d_2_d_2, d_2_d_3],
            [d_3_d_1, d_3_d_2, d_3_d_3]
        ])

        return max_jacobian_matrix1 @ max_jacobian_matrix2
