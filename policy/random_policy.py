"""
Random Policy for Balatro Dataset Generation

This policy selects actions completely randomly from the available action space.
Useful for baseline comparisons and exploring the full state space.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RandomPolicy:
    """
    Pure random policy that selects actions uniformly at random.
    """
    
    def __init__(self, action_space_size: int = 10, seed: Optional[int] = None):
        """
        Initialize random policy.
        
        Args:
            action_space_size: Number of possible actions in the environment
            seed: Random seed for reproducibility
        """
        self.action_space_size = action_space_size
        self.rng = np.random.RandomState(seed)
        self.total_actions = 0
        self.action_counts = np.zeros(action_space_size)
        
        logger.info(f"Initialized RandomPolicy with action space size {action_space_size}")
    
    def select_action(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        """
        Select a random action from the action space.
        
        Args:
            state: Current game state (unused in random policy)
            info: Additional game information (unused in random policy)
            
        Returns:
            Random action index
        """
        action = self.rng.randint(0, self.action_space_size)
        
        # Track action statistics
        self.total_actions += 1
        self.action_counts[action] += 1
        
        return action
    
    def get_action_distribution(self) -> Dict[int, float]:
        """
        Get the empirical action distribution.
        
        Returns:
            Dictionary mapping action to probability
        """
        if self.total_actions == 0:
            return {i: 0.0 for i in range(self.action_space_size)}
        
        return {
            i: count / self.total_actions 
            for i, count in enumerate(self.action_counts)
        }
    
    def reset_stats(self):
        """Reset action statistics."""
        self.total_actions = 0
        self.action_counts = np.zeros(self.action_space_size)
    
    def get_policy_info(self) -> Dict[str, Any]:
        """
        Get information about this policy.
        
        Returns:
            Policy metadata
        """
        return {
            'policy_type': 'random',
            'action_space_size': self.action_space_size,
            'total_actions': self.total_actions,
            'action_distribution': self.get_action_distribution()
        }


class BiasedRandomPolicy(RandomPolicy):
    """
    Random policy with configurable action biases.
    Useful for testing specific action preferences.
    """
    
    def __init__(self, action_space_size: int = 10, action_weights: Optional[np.ndarray] = None, seed: Optional[int] = None):
        """
        Initialize biased random policy.
        
        Args:
            action_space_size: Number of possible actions
            action_weights: Weights for each action (will be normalized)
            seed: Random seed for reproducibility
        """
        super().__init__(action_space_size, seed)
        
        if action_weights is not None:
            if len(action_weights) != action_space_size:
                raise ValueError(f"action_weights length {len(action_weights)} != action_space_size {action_space_size}")
            
            # Normalize weights to probabilities
            self.action_probs = action_weights / np.sum(action_weights)
        else:
            # Uniform distribution
            self.action_probs = np.ones(action_space_size) / action_space_size
        
        logger.info(f"Initialized BiasedRandomPolicy with weights: {self.action_probs}")
    
    def select_action(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        """
        Select action according to the configured probability distribution.
        
        Args:
            state: Current game state (unused)
            info: Additional game information (unused)
            
        Returns:
            Action sampled from the probability distribution
        """
        action = self.rng.choice(self.action_space_size, p=self.action_probs)
        
        # Track statistics
        self.total_actions += 1
        self.action_counts[action] += 1
        
        return action
    
    def get_policy_info(self) -> Dict[str, Any]:
        """Get policy information including configured weights."""
        info = super().get_policy_info()
        info.update({
            'policy_type': 'biased_random',
            'configured_probs': self.action_probs.tolist()
        })
        return info


class StateAwareRandomPolicy(RandomPolicy):
    """
    Random policy that adapts action probabilities based on game state.
    Provides more realistic random behavior.
    """
    
    def __init__(self, action_space_size: int = 10, seed: Optional[int] = None):
        super().__init__(action_space_size, seed)
        
        # Define action mappings based on balatro-gym
        self.ACTION_PLAY_HAND = 0
        self.ACTION_DISCARD_HAND = 1
        self.ACTION_SELECT_CARDS = list(range(2, 10))  # Actions 2-9 for card selection
        
    def select_action(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        """
        Select action with state-aware probabilities.
        
        Args:
            state: Current game state
            info: Additional game information
            
        Returns:
            Action selected based on state-aware probabilities
        """
        # Get basic state information
        selected_cards = state.get('selected_cards', [])
        discards_left = state.get('discards', 0)
        hands_left = state.get('hands_left', 0)
        money = state.get('money', 0)
        phase = info.get('phase', 'play')
        
        # Calculate action probabilities based on state
        action_probs = np.zeros(self.action_space_size)
        
        if phase == 'shop':
            # In shop phase, focus on shop actions (if available)
            # For now, default to card selection
            action_probs[self.ACTION_SELECT_CARDS] = 0.1
            action_probs[self.ACTION_PLAY_HAND] = 0.1
            action_probs[self.ACTION_DISCARD_HAND] = 0.1
        else:
            # In play phase
            if len(selected_cards) > 0:
                # If cards are selected, higher chance to play
                action_probs[self.ACTION_PLAY_HAND] = 0.6
                action_probs[self.ACTION_DISCARD_HAND] = 0.1
                action_probs[self.ACTION_SELECT_CARDS] = 0.05
            elif discards_left > 0:
                # If no cards selected and discards available
                action_probs[self.ACTION_PLAY_HAND] = 0.1
                action_probs[self.ACTION_DISCARD_HAND] = 0.3
                action_probs[self.ACTION_SELECT_CARDS] = 0.1
            else:
                # Must play or select cards
                action_probs[self.ACTION_PLAY_HAND] = 0.3
                action_probs[self.ACTION_DISCARD_HAND] = 0.0
                action_probs[self.ACTION_SELECT_CARDS] = 0.15
        
        # Normalize probabilities
        if np.sum(action_probs) == 0:
            action_probs = np.ones(self.action_space_size) / self.action_space_size
        else:
            action_probs = action_probs / np.sum(action_probs)
        
        # Sample action
        action = self.rng.choice(self.action_space_size, p=action_probs)
        
        # Track statistics
        self.total_actions += 1
        self.action_counts[action] += 1
        
        return action
    
    def get_policy_info(self) -> Dict[str, Any]:
        """Get policy information."""
        info = super().get_policy_info()
        info['policy_type'] = 'state_aware_random'
        return info


# Factory function for easy policy creation
def create_random_policy(policy_type: str = "pure", **kwargs) -> RandomPolicy:
    """
    Factory function to create different types of random policies.
    
    Args:
        policy_type: Type of random policy ('pure', 'biased', 'state_aware')
        **kwargs: Additional arguments passed to policy constructor
        
    Returns:
        Random policy instance
    """
    if policy_type == "pure":
        return RandomPolicy(**kwargs)
    elif policy_type == "biased":
        return BiasedRandomPolicy(**kwargs)
    elif policy_type == "state_aware":
        return StateAwareRandomPolicy(**kwargs)
    else:
        raise ValueError(f"Unknown random policy type: {policy_type}")


if __name__ == "__main__":
    # Example usage and testing
    import json
    
    # Test pure random policy
    print("Testing Pure Random Policy:")
    policy = RandomPolicy(action_space_size=10, seed=42)
    
    dummy_state = {'money': 10, 'selected_cards': []}
    dummy_info = {'phase': 'play'}
    
    # Generate some actions
    actions = []
    for _ in range(100):
        action = policy.select_action(dummy_state, dummy_info)
        actions.append(action)
    
    print(f"Generated {len(actions)} actions")
    print(f"Action distribution: {policy.get_action_distribution()}")
    print(f"Policy info: {json.dumps(policy.get_policy_info(), indent=2)}")
    
    print("\nTesting State-Aware Random Policy:")
    state_policy = StateAwareRandomPolicy(action_space_size=10, seed=42)
    
    # Test different states
    states = [
        ({'selected_cards': [], 'discards': 3, 'money': 10}, {'phase': 'play'}),
        ({'selected_cards': [0, 1], 'discards': 2, 'money': 5}, {'phase': 'play'}),
        ({'selected_cards': [], 'discards': 0, 'money': 15}, {'phase': 'play'}),
    ]
    
    for i, (state, info) in enumerate(states):
        actions = [state_policy.select_action(state, info) for _ in range(20)]
        print(f"State {i+1} actions: {actions}")
    
    print(f"State-aware policy info: {json.dumps(state_policy.get_policy_info(), indent=2)}")
