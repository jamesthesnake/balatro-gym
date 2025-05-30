import gymnasium as gym
from balatro_gym.env import BalatroEnv
from balatro_gym.wrappers.trajectory_logger import TrajectoryLogger
import random

NUM_EPISODES = 1000

def simple_policy(obs, action_mask):
    legal_actions = [i for i, valid in enumerate(action_mask) if valid]
    return random.choice(legal_actions)

env = gym.make("Balatro-v0")
logger = TrajectoryLogger()

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    while not done:
        action = simple_policy(obs, obs.get("action_mask", [1]*10))
        obs, reward, done, info = env.step(action)
        logger.log_step(obs, action, reward, done, info)
    logger.save_episode(episode)
    print(f"✅ Episode {episode} saved")
