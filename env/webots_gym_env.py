import gymnasium as gym
from gymnasium import spaces
import numpy as np

class WebotsCartPoleEnv(gym.Env):
    def __init__(self):
        super(WebotsCartPoleEnv, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        high = np.array([2.4, 10.0, np.pi, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None
        self.time_step = 0
        self.max_steps = 500

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([
            self.np_random.uniform(-0.05, 0.05),  # cart pos
            0.0,
            self.np_random.uniform(-0.05, 0.05),  # pole angle
            0.0
        ], dtype=np.float32)
        self.time_step = 0
        return self.state, {}

    def step(self, action):
        force = np.clip(action[0], -1.0, 1.0) * 10.0
        x, x_dot, theta, theta_dot = self.state

        gravity = 9.8
        mass_cart = 1.0
        mass_pole = 0.1
        total_mass = mass_cart + mass_pole
        length = 0.5
        polemass_length = mass_pole * length
        tau = 0.02

        temp = (force + polemass_length * theta_dot**2 * np.sin(theta)) / total_mass
        theta_acc = (gravity * np.sin(theta) - np.cos(theta) * temp) / (
            length * (4.0 / 3.0 - mass_pole * np.cos(theta)**2 / total_mass))
        x_acc = temp - polemass_length * theta_acc * np.cos(theta) / total_mass

        x += tau * x_dot
        x_dot += tau * x_acc
        theta += tau * theta_dot
        theta_dot += tau * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.time_step += 1

        terminated = bool(x < -2.4 or x > 2.4 or theta < -np.pi/2 or theta > np.pi/2)
        truncated = self.time_step >= self.max_steps

        reward = 1.0 - (theta**2 + 0.1 * theta_dot**2 + 0.01 * x**2 + 0.01 * x_dot**2)
        reward = float(reward)
        if terminated or truncated:
            reward = -10.0

        return self.state, reward, terminated, truncated, {}
