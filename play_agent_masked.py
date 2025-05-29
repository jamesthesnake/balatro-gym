import balatro_gym  # Registers the env
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
from action_mask_wrapper import ActionMaskWrapper

# Create env
env = gym.make("BalatroSmall-v0", chip_threshold=250, render_mode="ansi")
env = FlattenObservation(env)
env = ActionMaskWrapper(env)

obs, _ = env.reset()
obs_vec = obs["observation"]
mask = obs["action_mask"]

done = False
total_reward = 0

while not done:
    print(env.render())
    
    # Pick a random valid action
    action = np.random.choice(np.where(mask == 1)[0])
    print(f"Agent chose action: {action}")

    obs, reward, done, truncated, info = env.step(action)
    obs_vec = obs["observation"]
    mask = obs["action_mask"]
    total_reward += reward

print(f"🎉 Game over. Total reward: {total_reward}")
env.close()

