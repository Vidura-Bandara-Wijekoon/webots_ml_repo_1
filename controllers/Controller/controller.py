from controller import Robot
import torch
import numpy as np
from stable_baselines3 import PPO

# Load trained model
model = PPO.load("ppo_cartpole")

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Devices
motor = robot.getDevice("horizontal_motor")
position_sensor = robot.getDevice("horizontal position sensor")
hip_sensor = robot.getDevice("hip")

position_sensor.enable(timestep)
hip_sensor.enable(timestep)

motor.setPosition(float('inf'))

# State memory
prev_angle = 0
prev_pos = 0

while robot.step(timestep) != -1:
    # State: [cart_pos, cart_vel, pendulum_angle, pendulum_angular_vel]
    cart_pos = position_sensor.getValue()
    pend_angle = hip_sensor.getValue()

    cart_vel = (cart_pos - prev_pos) / (timestep / 1000.0)
    pend_vel = (pend_angle - prev_angle) / (timestep / 1000.0)

    state = np.array([cart_pos, cart_vel, pend_angle, pend_vel], dtype=np.float32)

    action, _ = model.predict(state, deterministic=True)

    # Map action to force (if continuous)
    force = float(action)
    motor.setVelocity(force)

    # Store previous values
    prev_pos = cart_pos
    prev_angle = pend_angle
