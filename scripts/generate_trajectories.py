import gymnasium as gym
from balatro_gym.env import BalatroEnv # Assuming BalatroEnv is directly importable
from balatro_gym.wrappers.trajectory_logger import TrajectoryLogger
import random
import numpy as np # action_mask might be a numpy array

NUM_EPISODES = 1000

def simple_policy(obs, action_mask):
    # Ensure action_mask is treated as a list or numpy array for consistent iteration
    if isinstance(action_mask, np.ndarray):
        action_mask_list = action_mask.tolist()
    else:
        action_mask_list = action_mask

    print(f"Observation keys: {obs.keys()}")
    print(f"Received action_mask: {action_mask_list}") # Print the list version

    legal_actions = [i for i, x in enumerate(action_mask_list) if x == 1]
    print(f"Legal actions before choice: {legal_actions}")

    if not legal_actions:
        print("WARNING: No legal actions found! This often means the game is over, stuck, or not running correctly.")
        # Instead of raising an error, you might want a more graceful exit or
        # a "no-op" action if the environment supports it, or a specific handling
        # for end-of-game states.
        # For now, we'll keep the raise to confirm the issue.
        raise IndexError('Cannot choose from an empty sequence because legal_actions is empty.')

    return random.choice(legal_actions)

# --- Main execution ---
if __name__ == "__main__":
    env = gym.make("Balatro-v0")
    # You might want to wrap the environment with the logger after creation
    # or ensure the logger is correctly integrated.
    # The TrajectoryLogger expects env, so it should be passed like this:
    # logger = TrajectoryLogger(env) # assuming TrajectoryLogger takes env
    # Or, if it's just logging steps:
    logger = TrajectoryLogger() # as you have it, which means it relies on manual logging

    for episode in range(NUM_EPISODES):
        print(f"\n--- Starting Episode {episode} ---")
        obs, info = env.reset() # env.reset() returns (observation, info)

        done = False
        truncated = False # Initialize truncated flag for gymnasium new API

        # This loop continues as long as the episode is not done AND not truncated
        while not done and not truncated:
            # The action_mask is part of the 'info' dictionary in newer Gymnasium versions
            # or part of the 'obs' dictionary in some custom environments.
            # Your current code `obs.get("action_mask", [1]*10)` suggests it's in obs.
            # We'll stick to that, but be aware of env.step() returns.

            # Get action_mask from observation
            action_mask = obs.get("action_mask", [0]*env.action_space.n) # Use env.action_space.n for default size

            # Handle the case where action_mask might be a numpy array of bools
            if isinstance(action_mask, np.ndarray) and action_mask.dtype == bool:
                action_mask = action_mask.astype(int).tolist() # Convert bool array to int list

            # If action_mask is still a numpy array of 0s and 1s
            elif isinstance(action_mask, np.ndarray):
                action_mask = action_mask.tolist()


            action = simple_policy(obs, action_mask)

            # env.step returns (observation, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = env.step(action)

            # In Gymnasium, 'done' is now typically 'terminated or truncated'
            done = terminated or truncated

            logger.log_step(obs, action, reward, terminated, info) # Log terminated, not general done

            if done: # If the episode is over, print and break
                print(f"✅ Episode {episode} finished (terminated={terminated}, truncated={truncated})")
                break # Exit the inner while loop to go to the next episode

        logger.save_episode(episode) # Save after the episode concludes
        print(f"✅ Episode {episode} data saved")

    env.close() # Close the environment when done
