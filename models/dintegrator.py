import numpy as np

class DoubleIntegrator:
    def __init__(self, time_step=1, k=1e-5):
        self.time_step = time_step
        self.k = k
        self.pos = 0.0
        self.vel = 0.0

    def set_state(self, pos, vel):
        self.pos = pos
        self.vel = vel

        return self

    def get_state(self):
        return self.pos, self.vel
    
    def update_dynamics(self, control):

        for u in control:
            new_pos = self.pos + self.time_step * self.vel + 0.5 * self.time_step ** 2 * u
            new_vel = self.vel - self.k * self.time_step * self.vel**3 + self.time_step * u

            self.pos = new_pos
            self.vel = new_vel
        
        return self.get_state()
    
    def max_jacobian(self, state, control):
        # Partial derivatives of the dynamics equations with respect to angle and angular velocity
        d_1_d_1 = 1
        d_1_d_2 = self.time_step
        d_2_d_1 = 0
        d_2_d_2 = 1 + 3 * self.k * self.time_step * (abs(state[1])+1)**2

        max_jacobian_matrix = np.array([
            [d_1_d_1, d_1_d_2],
            [d_2_d_1, d_2_d_2]
        ])

        return np.linalg.matrix_power(max_jacobian_matrix, 2)
