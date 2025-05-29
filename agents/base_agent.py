from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BaseAgent(ABC):
    """Abstract base class for Balatro agents."""
    
    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        self.episode_count = 0
        self.total_reward = 0.0
    
    @abstractmethod
    def get_action(self, observation: Dict[str, np.ndarray], 
                   action_mask: np.ndarray, env) -> int:
        """
        Select an action given the current observation and legal actions.
        
        Args:
            observation: Current game state observation
            action_mask: Binary mask of legal actions
            env: Environment instance (for additional context)
        
        Returns:
            action_idx: Index of chosen action
        """
        pass
    
    def on_episode_start(self):
        """Called at the start of each episode."""
        self.episode_count += 1
    
    def on_episode_end(self, total_reward: float, info: Dict[str, Any]):
        """Called at the end of each episode."""
        self.total_reward += total_reward
    
    def save(self, path: str):
        """Save agent state to disk."""
        pass
    
    def load(self, path: str):
        """Load agent state from disk."""
        pass
