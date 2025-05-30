"""
Shared data structures for Balatro dataset generation and viewing.
This module contains all the common data classes used across the project.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import json

@dataclass
class TrajectoryStep:
    """
    Represents a single step in a Balatro trajectory.
    
    Attributes:
        state: Game state before taking the action
        action: Action taken (integer index)
        reward: Immediate reward received
        next_state: Game state after taking the action
        done: Whether the episode ended
        info: Additional information from the environment
        rtg: Reward-to-go (cumulative future reward)
        strategy_tags: List of strategy labels for this step
        metadata: Additional metadata about this step
    """
    state: Dict[str, Any]
    action: int
    reward: float
    next_state: Dict[str, Any]
    done: bool
    info: Dict[str, Any]
    rtg: float
    strategy_tags: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrajectoryStep':
        """Create TrajectoryStep from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TrajectoryStep':
        """Create TrajectoryStep from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

@dataclass
class EpisodeMetadata:
    """
    Metadata for an episode.
    """
    episode_id: int
    worker_id: int
    policy_type: str
    total_steps: int
    total_reward: float
    episode_length: int
    start_timestamp: float
    end_timestamp: float
    final_ante: int = 0
    max_chips: float = 0.0
    max_money: float = 0.0
    strategy_tag_counts: Dict[str, int] = None
    
    def __post_init__(self):
        if self.strategy_tag_counts is None:
            self.strategy_tag_counts = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

@dataclass
class DatasetMetadata:
    """
    Metadata for the entire dataset.
    """
    total_episodes: int
    total_steps: int
    total_reward: float
    avg_steps_per_episode: float
    avg_reward_per_episode: float
    policy_type: str
    generation_timestamp: float
    num_workers: int
    dataset_version: str = "1.0"
    balatro_gym_version: str = "unknown"
    generation_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.generation_config is None:
            self.generation_config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

# Action constants for consistency across the project
class BalatroActions:
    """Constants for Balatro action space."""
    PLAY_HAND = 0
    DISCARD_HAND = 1
    SELECT_CARD_1 = 2
    SELECT_CARD_2 = 3
    SELECT_CARD_3 = 4
    SELECT_CARD_4 = 5
    SELECT_CARD_5 = 6
    SELECT_CARD_6 = 7
    SELECT_CARD_7 = 8
    SELECT_CARD_8 = 9
    
    # Action name mapping
    ACTION_NAMES = {
        PLAY_HAND: "play_hand",
        DISCARD_HAND: "discard_hand",
        SELECT_CARD_1: "select_card_1",
        SELECT_CARD_2: "select_card_2",
        SELECT_CARD_3: "select_card_3",
        SELECT_CARD_4: "select_card_4",
        SELECT_CARD_5: "select_card_5",
        SELECT_CARD_6: "select_card_6",
        SELECT_CARD_7: "select_card_7",
        SELECT_CARD_8: "select_card_8",
    }
    
    @classmethod
    def get_action_name(cls, action: int) -> str:
        """Get human-readable action name."""
        return cls.ACTION_NAMES.get(action, f"unknown_action_{action}")
    
    @classmethod
    def get_action_list(cls) -> List[int]:
        """Get list of all valid actions."""
        return list(cls.ACTION_NAMES.keys())

# Strategy tag constants
class StrategyTags:
    """Common strategy tags used throughout the system."""
    
    # Game phases
    ANTE_START = "ante_start"
    BLIND_SELECTION = "blind_selection"
    SHOP_PHASE = "shop_phase"
    BOSS_PREP = "boss_prep"
    
    # Action types
    PLAY_HAND = "action_play_hand"
    DISCARD_HAND = "action_discard_hand"
    SELECT_CARD = "action_select_card"
    
    # Strategic patterns
    HAND_BUILDING = "hand_building"
    ECONOMY_FOCUS = "economy_focus"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    HIGH_RISK = "high_risk"
    SAFE_PLAY = "safe_play"
    
    # Hand types
    HIGH_CARD = "hand_high_card"
    PAIR = "hand_pair"
    TWO_PAIR = "hand_two_pair"
    THREE_OF_A_KIND = "hand_three_of_a_kind"
    STRAIGHT = "hand_straight"
    FLUSH = "hand_flush"
    FULL_HOUSE = "hand_full_house"
    FOUR_OF_A_KIND = "hand_four_of_a_kind"
    STRAIGHT_FLUSH = "hand_straight_flush"
    ROYAL_FLUSH = "hand_royal_flush"
    
    @classmethod
    def get_all_tags(cls) -> List[str]:
        """Get all defined strategy tags."""
        return [
            value for name, value in cls.__dict__.items()
            if isinstance(value, str) and not name.startswith('_')
        ]

# Utility functions for data handling
def serialize_state(state) -> Dict[str, Any]:
    """
    Serialize game state for storage.
    
    Args:
        state: Game state from environment
        
    Returns:
        Serializable dictionary representation
    """
    if isinstance(state, dict):
        return state.copy()
    elif hasattr(state, '__dict__'):
        return vars(state)
    else:
        return {"raw_state": str(state)}

def serialize_info(info) -> Dict[str, Any]:
    """
    Serialize info dictionary for storage.
    
    Args:
        info: Info dictionary from environment
        
    Returns:
        Serializable dictionary representation
    """
    if isinstance(info, dict):
        return info.copy()
    elif hasattr(info, '__dict__'):
        return vars(info)
    else:
        return {"raw_info": str(info)}

def validate_trajectory_step(step: TrajectoryStep) -> bool:
    """
    Validate that a trajectory step has all required fields.
    
    Args:
        step: TrajectoryStep to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required fields exist and have correct types
        assert isinstance(step.state, dict), "state must be a dictionary"
        assert isinstance(step.action, int), "action must be an integer"
        assert isinstance(step.reward, (int, float)), "reward must be numeric"
        assert isinstance(step.next_state, dict), "next_state must be a dictionary"
        assert isinstance(step.done, bool), "done must be boolean"
        assert isinstance(step.info, dict), "info must be a dictionary"
        assert isinstance(step.rtg, (int, float)), "rtg must be numeric"
        assert isinstance(step.strategy_tags, list), "strategy_tags must be a list"
        assert isinstance(step.metadata, dict), "metadata must be a dictionary"
        
        # Check action is in valid range
        valid_actions = BalatroActions.get_action_list()
        if step.action not in valid_actions:
            print(f"Warning: action {step.action} not in valid action list")
        
        return True
        
    except (AssertionError, AttributeError) as e:
        print(f"Trajectory step validation failed: {e}")
        return False

def create_empty_trajectory_step() -> TrajectoryStep:
    """Create an empty trajectory step with default values."""
    return TrajectoryStep(
        state={},
        action=0,
        reward=0.0,
        next_state={},
        done=False,
        info={},
        rtg=0.0,
        strategy_tags=[],
        metadata={}
    )

# Episode utility functions
def compute_episode_statistics(episode: List[TrajectoryStep]) -> Dict[str, Any]:
    """
    Compute statistics for an episode.
    
    Args:
        episode: List of trajectory steps
        
    Returns:
        Dictionary of episode statistics
    """
    if not episode:
        return {}
    
    total_reward = sum(step.reward for step in episode)
    total_steps = len(episode)
    
    # Action distribution
    action_counts = {}
    for step in episode:
        action = step.action
        action_counts[action] = action_counts.get(action, 0) + 1
    
    # Strategy tag distribution
    strategy_counts = {}
    for step in episode:
        for tag in step.strategy_tags:
            strategy_counts[tag] = strategy_counts.get(tag, 0) + 1
    
    # Reward statistics
    rewards = [step.reward for step in episode]
    avg_reward = total_reward / total_steps if total_steps > 0 else 0
    max_reward = max(rewards) if rewards else 0
    min_reward = min(rewards) if rewards else 0
    
    return {
        'total_reward': total_reward,
        'total_steps': total_steps,
        'avg_reward_per_step': avg_reward,
        'max_reward': max_reward,
        'min_reward': min_reward,
        'action_distribution': action_counts,
        'strategy_tag_counts': strategy_counts,
        'final_done': episode[-1].done if episode else False
    }

# Dataset utility functions
def validate_dataset_consistency(episodes: List[List[TrajectoryStep]]) -> Dict[str, Any]:
    """
    Validate consistency across a dataset.
    
    Args:
        episodes: List of episodes (each episode is a list of trajectory steps)
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_episodes': len(episodes),
        'valid_episodes': 0,
        'invalid_episodes': 0,
        'total_steps': 0,
        'validation_errors': []
    }
    
    for i, episode in enumerate(episodes):
        episode_valid = True
        
        if not episode:
            validation_results['validation_errors'].append(f"Episode {i} is empty")
            episode_valid = False
        else:
            validation_results['total_steps'] += len(episode)
            
            # Validate each step
            for j, step in enumerate(episode):
                if not validate_trajectory_step(step):
                    validation_results['validation_errors'].append(
                        f"Episode {i}, step {j} failed validation"
                    )
                    episode_valid = False
        
        if episode_valid:
            validation_results['valid_episodes'] += 1
        else:
            validation_results['invalid_episodes'] += 1
    
    return validation_results
