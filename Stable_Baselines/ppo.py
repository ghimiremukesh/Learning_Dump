# train a ppo agent using stable-baselines environment

import gym
from stable_baselines3 import PPO

# create an environment
env = gym.make("Lunar")

# build PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)

# see how it learned
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
