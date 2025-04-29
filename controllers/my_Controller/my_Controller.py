from controller import Robot
import torch
import numpy as np
from stable_baselines3 import PPO

model = PPO.load("E:/test_webot/inverted_pendulum/models/ppo_cartpole")

robot = Robot()
timestep = int(robot.getBasicTimeStep())

motor = robot.getDevice("horizontal_motor")
position_sensor = robot.getDevice("horizontal position sensor")
hip_sensor = robot.getDevice("hip")

position_sensor.enable(timestep)
hip_sensor.enable(timestep)

motor.setPosition(float('inf'))   # velocity control mode
motor.setVelocity(0.0)            # initialize

prev_angle = 0
prev_pos = 0

while robot.step(timestep) != -1:
    cart_pos = position_sensor.getValue()
    pend_angle = hip_sensor.getValue()

    cart_vel = (cart_pos - prev_pos) / (timestep / 1000.0)
    pend_vel = (pend_angle - prev_angle) / (timestep / 1000.0)

    state = np.array([cart_pos, cart_vel, pend_angle, pend_vel], dtype=np.float32)
    action, _ = model.predict(state, deterministic=True)

    velocity_command = float(action[0]) * 6.0  # scale factor
    motor.setVelocity(velocity_command)

    prev_pos = cart_pos
    prev_angle = pend_angle
    print(f"Cart Position: {cart_pos}, Pendulum Angle: {pend_angle}, Action: {action[0]}")