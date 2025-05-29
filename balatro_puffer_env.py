import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pufferlib
import pufferlib.emulation
from gymnasium.wrappers import FlattenObservation

import balatro_gym  # Ensure it registers "BalatroSmall-v0"

# ==== Wrapper to Add Action Mask ====
class ActionMaskWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.orig_obs_space = env.observation_space

        self.action_mask_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.env.action_space.n,),
            dtype=np.int8
        )

        self.observation_space = spaces.Dict({
            "observation": self.orig_obs_space,
            "action_mask": self.action_mask_space
        })

    def observation(self, obs):
        valid = self.env.get_wrapper_attr("valid_actions")()
        mask = np.zeros(self.env.action_space.n, dtype=np.int8)
        mask[valid] = 1
        return {
            "observation": obs,
            "action_mask": mask
        }

# ==== PufferLib-Compatible Environment ====
class BalatroPufferEnv(pufferlib.emulation.GymnasiumPufferEnv):
    def __init__(self):
        super().__init__(
            env_creator=lambda: gym.make("BalatroSmall-v0", chip_threshold=250, render_mode="ansi"),
            wrappers=[
                FlattenObservation,
                ActionMaskWrapper,
            ]
        )

