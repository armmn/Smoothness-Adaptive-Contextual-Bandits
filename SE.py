from utilities import get_u


class SE:
    def __init__(self, T, gamma):
        self.T = T
        self.gamma = gamma

        # initialization
        self.tau = 1
        self.S = [0, 1]
        self.Y_bar = [0, 0]
        self.Y_bar_max = 0
        self.arm_idx = -1

    def collect_observation(self, y, arm):
        self.Y_bar[arm] = ((self.tau - 1) * self.Y_bar[arm] + y) / self.tau
        self.Y_bar_max = max(self.Y_bar)

    def get_arm(self):
        self.arm_idx += 1
        while self.arm_idx < len(self.S):
            arm = self.S[self.arm_idx]
            if self.Y_bar[arm] >= self.Y_bar_max - self.gamma * get_u(self.tau, self.T):
                return arm, self.tau, self.S
            else:
                self.S.pop(self.arm_idx)
        self.tau += 1
        self.arm_idx = 0
        while self.arm_idx < len(self.S):
            arm = self.S[self.arm_idx]
            if self.Y_bar[arm] >= self.Y_bar_max - self.gamma * get_u(self.tau, self.T):
                return arm, self.tau, self.S
            else:
                self.S.pop(self.arm_idx)
