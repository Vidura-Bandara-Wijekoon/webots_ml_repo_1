# train_rl_model.py 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'env'))

from webots_gym_env import WebotsCartPoleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = WebotsCartPoleEnv()
check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

model.save("../models/ppo_cartpole")
print("✅ Model saved to /models/ppo_cartpole.zip")
