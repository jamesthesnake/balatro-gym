import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ActionMaskWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.orig_obs_space = env.observation_space

        # Action mask: binary vector of valid actions
        self.action_mask_space = spaces.Box(0, 1, shape=(self.env.action_space.n,), dtype=np.int8)

        self.observation_space = spaces.Dict({
            "observation": self.orig_obs_space,
            "action_mask": self.action_mask_space
        })

    def observation(self, obs):
        valid = self.env.get_wrapper_attr('valid_actions')()
        mask = np.zeros(self.env.action_space.n, dtype=np.int8)
        mask[valid] = 1
        return {
            "observation": obs,
            "action_mask": mask
        }

