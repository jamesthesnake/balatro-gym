#!/usr/bin/env python3
"""
Balatro Gym Trajectory Recording Script

This script records trajectories while performing different actions in the Balatro Gym environment.
It supports various action strategies and saves detailed trajectory data for analysis.
"""

import gymnasium as gym
import numpy as np
import json
import pickle
import time
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import argparse
import logging

# Import Balatro-specific components if available
try:
    from balatro_gym import Action as BalatroAction
    BALATRO_GYM_AVAILABLE = True
except ImportError:
    # Fallback action enumeration if balatro_gym not installed
    class BalatroAction(Enum):
        """Fallback Balatro action enumeration"""
        PLAY_CARD_0 = 0
        PLAY_CARD_1 = 1
        PLAY_CARD_2 = 2
        PLAY_CARD_3 = 3
        PLAY_CARD_4 = 4
        DISCARD_HAND = 5
        END_HAND = 6
        SKIP_BLIND = 7
        USE_CONSUMABLE = 8
        BUY_JOKER_SLOT_0 = 9
        BUY_JOKER_SLOT_1 = 10
        BUY_JOKER_SLOT_2 = 11
        BUY_JOKER_SLOT_3 = 12
        BUY_JOKER_SLOT_4 = 13
        BUY_PACK_TAROT = 14
        BUY_PACK_PLANET = 15
        BUY_PACK_SPECTRAL = 16
        BUY_VOUCHER = 17
        NO_OP = 18
    BALATRO_GYM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrajectoryStep:
    """Represents a single step in a trajectory"""
    step_num: int
    observation: Dict
    action: int
    action_name: str
    reward: float
    terminated: bool
    truncated: bool
    info: Dict
    timestamp: float

@dataclass
class Trajectory:
    """Represents a complete trajectory/episode"""
    episode_id: str
    steps: List[TrajectoryStep]
    total_reward: float
    episode_length: int
    start_time: float
    end_time: float
    strategy_name: str
    metadata: Dict

class ActionStrategy:
    """Base class for action strategies"""
    def __init__(self, name: str):
        self.name = name
    
    def select_action(self, observation, action_space, step_num: int) -> int:
        """Select an action based on the strategy"""
        raise NotImplementedError

class RandomStrategy(ActionStrategy):
    """Random action strategy"""
    def __init__(self, seed: Optional[int] = None):
        super().__init__("random")
        self.rng = np.random.RandomState(seed)
    
    def select_action(self, observation, action_space, step_num: int) -> int:
        return self.rng.choice(action_space.n)

class BiasedStrategy(ActionStrategy):
    """Strategy that favors certain actions"""
    def __init__(self, action_weights: Dict[int, float], seed: Optional[int] = None):
        super().__init__("biased")
        self.action_weights = action_weights
        self.rng = np.random.RandomState(seed)
    
    def select_action(self, observation, action_space, step_num: int) -> int:
        # Create probability distribution based on weights
        probs = np.ones(action_space.n)
        for action, weight in self.action_weights.items():
            if action < action_space.n:
                probs[action] = weight
        
        probs = probs / np.sum(probs)  # Normalize
        return self.rng.choice(action_space.n, p=probs)

class SequentialStrategy(ActionStrategy):
    """Strategy that cycles through actions sequentially"""
    def __init__(self, action_sequence: List[int]):
        super().__init__("sequential")
        self.action_sequence = action_sequence
    
    def select_action(self, observation, action_space, step_num: int) -> int:
        return self.action_sequence[step_num % len(self.action_sequence)]

class BalatroSpecificStrategy(ActionStrategy):
    """Strategy that understands Balatro game mechanics"""
    def __init__(self, seed: Optional[int] = None):
        super().__init__("balatro_specific")
        self.rng = np.random.RandomState(seed)
    
    def select_action(self, observation, action_space, step_num: int) -> int:
        # Extract game state from observation
        money = observation.get('money', [0])[0] if isinstance(observation, dict) else 0
        hands_remaining = observation.get('hands_remaining', [0])[0] if isinstance(observation, dict) else 0
        discards_remaining = observation.get('discards_remaining', [0])[0] if isinstance(observation, dict) else 0
        in_shop = observation.get('in_shop', [0])[0] if isinstance(observation, dict) else 0
        action_mask = observation.get('action_mask', np.ones(action_space.n)) if isinstance(observation, dict) else np.ones(action_space.n)
        
        # Get valid actions based on action mask
        valid_actions = [i for i in range(action_space.n) if action_mask[i] == 1]
        
        if not valid_actions:
            return 0  # Fallback to first action if no valid actions
        
        # Balatro-specific logic
        if in_shop:
            # In shop: prefer buying jokers > packs > vouchers, but consider money
            if money >= 10:  # Enough for most purchases
                shop_actions = [a for a in valid_actions if 9 <= a <= 17]  # BUY_JOKER_SLOT_* and BUY_PACK_*
                if shop_actions:
                    return self.rng.choice(shop_actions)
            return self.rng.choice(valid_actions)
        else:
            # Not in shop: prioritize playing cards when possible
            if hands_remaining > 0:
                play_actions = [a for a in valid_actions if 0 <= a <= 4]  # PLAY_CARD_*
                if play_actions and self.rng.random() < 0.7:
                    return self.rng.choice(play_actions)
            
            # Use discards strategically
            if discards_remaining > 0 and hands_remaining > 1:
                discard_action = 5  # DISCARD_HAND
                if discard_action in valid_actions and self.rng.random() < 0.3:
                    return discard_action
            
            return self.rng.choice(valid_actions)

class ShopFocusedStrategy(ActionStrategy):
    """Strategy that focuses on shop actions when available"""
    def __init__(self, seed: Optional[int] = None):
        super().__init__("shop_focused")
        self.rng = np.random.RandomState(seed)
    
    def select_action(self, observation, action_space, step_num: int) -> int:
        in_shop = observation.get('in_shop', [0])[0] if isinstance(observation, dict) else 0
        action_mask = observation.get('action_mask', np.ones(action_space.n)) if isinstance(observation, dict) else np.ones(action_space.n)
        valid_actions = [i for i in range(action_space.n) if action_mask[i] == 1]
        
        if not valid_actions:
            return 0
        
        if in_shop:
            # Strongly prefer shop actions
            return self.rng.choice(valid_actions)
        else:
            # Try to get to shop quickly - prefer ending hands/skipping
            end_actions = [a for a in valid_actions if a in [6, 7]]  # END_HAND, SKIP_BLIND
            if end_actions and self.rng.random() < 0.5:
                return self.rng.choice(end_actions)
            return self.rng.choice(valid_actions)

class ConservativeStrategy(ActionStrategy):
    """Conservative strategy focusing on basic card play"""
    def __init__(self, seed: Optional[int] = None):
        super().__init__("conservative")
        self.rng = np.random.RandomState(seed)
    
    def select_action(self, observation, action_space, step_num: int) -> int:
        action_mask = observation.get('action_mask', np.ones(action_space.n)) if isinstance(observation, dict) else np.ones(action_space.n)
        valid_actions = [i for i in range(action_space.n) if action_mask[i] == 1]
        
        if not valid_actions:
            return 0
        
        # Favor basic play/discard actions
        basic_actions = [a for a in valid_actions if a <= 5]  # PLAY_CARD_* and DISCARD_HAND
        if basic_actions and self.rng.random() < 0.8:
            return self.rng.choice(basic_actions)
        
        return self.rng.choice(valid_actions)

class AggressiveStrategy(ActionStrategy):
    """Aggressive strategy exploring all available actions"""
    def __init__(self, seed: Optional[int] = None):
        super().__init__("aggressive")
        self.rng = np.random.RandomState(seed)
    
    def select_action(self, observation, action_space, step_num: int) -> int:
        action_mask = observation.get('action_mask', np.ones(action_space.n)) if isinstance(observation, dict) else np.ones(action_space.n)
        valid_actions = [i for i in range(action_space.n) if action_mask[i] == 1]
        
        if not valid_actions:
            return 0
        
        # Prefer shop and consumable actions when available
        advanced_actions = [a for a in valid_actions if a >= 8]  # USE_CONSUMABLE and shop actions
        if advanced_actions and self.rng.random() < 0.6:
            return self.rng.choice(advanced_actions)
        
        return self.rng.choice(valid_actions)

class TrajectoryRecorder:
    """Main class for recording trajectories in Balatro Gym environment"""
    
    def __init__(self, env_id: str = "Balatro-v0", output_dir: str = "trajectories"):
        """
        Initialize the trajectory recorder
        
        Args:
            env_id: Gymnasium environment ID for Balatro
            output_dir: Directory to save trajectory data
        """
        self.env_id = env_id
        self.output_dir = output_dir
        self.env = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Action name mapping - use actual Balatro actions if available
        if BALATRO_GYM_AVAILABLE:
            try:
                from balatro_gym import Action
                self.action_names = {action.value: action.name for action in Action}
                logger.info("Using actual Balatro Gym action definitions")
            except ImportError:
                self.action_names = {action.value: action.name for action in BalatroAction}
                logger.warning("Using fallback action definitions")
        else:
            self.action_names = {action.value: action.name for action in BalatroAction}
            logger.warning("Balatro Gym not available, using fallback action definitions")
    
    def setup_environment(self, render_mode: Optional[str] = None):
        """Setup the Balatro Gym environment"""
        try:
            self.env = gym.make(self.env_id, render_mode=render_mode)
            logger.info(f"Environment {self.env_id} created successfully")
            logger.info(f"Action space: {self.env.action_space}")
            logger.info(f"Observation space: {self.env.observation_space}")
            
            # Log Balatro-specific information
            if hasattr(self.env, 'money'):
                logger.info(f"Initial money: {self.env.money}")
            if hasattr(self.env, 'current_ante'):
                logger.info(f"Initial ante: {self.env.current_ante}")
                
        except Exception as e:
            logger.error(f"Failed to create environment {self.env_id}: {e}")
            logger.error("Please ensure the Balatro Gym environment is properly installed.")
            logger.error("You may need to install it with: pip install balatro-gym")
            logger.error("Or if installing from source: pip install -e .")
            raise
    
    def get_action_name(self, action: int) -> str:
        """Get human-readable action name"""
        return self.action_names.get(action, f"Unknown_Action_{action}")
    
    def record_episode(self, strategy: ActionStrategy, max_steps: int = 1000, 
                      episode_id: Optional[str] = None) -> Trajectory:
        """
        Record a single episode using the given strategy
        
        Args:
            strategy: Action selection strategy
            max_steps: Maximum number of steps per episode
            episode_id: Unique identifier for the episode
            
        Returns:
            Recorded trajectory
        """
        if self.env is None:
            raise ValueError("Environment not setup. Call setup_environment() first.")
        
        if episode_id is None:
            episode_id = f"{strategy.name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        logger.info(f"Starting episode {episode_id} with strategy {strategy.name}")
        
        # Reset environment
        observation, info = self.env.reset()
        
        trajectory = Trajectory(
            episode_id=episode_id,
            steps=[],
            total_reward=0.0,
            episode_length=0,
            start_time=time.time(),
            end_time=0.0,
            strategy_name=strategy.name,
            metadata={
                "env_id": self.env_id,
                "max_steps": max_steps,
                "action_space_size": self.env.action_space.n
            }
        )
        
        step_num = 0
        while step_num < max_steps:
            # Select action using strategy
            action = strategy.select_action(observation, self.env.action_space, step_num)
            action_name = self.get_action_name(action)
            
            # Take step in environment
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            
            # Record step
            step = TrajectoryStep(
                step_num=step_num,
                observation=self._serialize_observation(observation),
                action=action,
                action_name=action_name,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                timestamp=time.time()
            )
            
            trajectory.steps.append(step)
            trajectory.total_reward += reward
            
            # Update for next iteration
            observation = next_observation
            step_num += 1
            
            # Check if episode is done
            if terminated or truncated:
                logger.info(f"Episode {episode_id} ended at step {step_num}: "
                          f"terminated={terminated}, truncated={truncated}")
                break
        
        trajectory.episode_length = len(trajectory.steps)
        trajectory.end_time = time.time()
        
        logger.info(f"Episode {episode_id} completed: {trajectory.episode_length} steps, "
                   f"total reward: {trajectory.total_reward:.2f}")
        
        return trajectory
    
    def _serialize_observation(self, observation) -> Dict:
        """Serialize observation for JSON storage"""
        if isinstance(observation, np.ndarray):
            return {"type": "numpy_array", "data": observation.tolist(), "shape": list(observation.shape)}
        elif isinstance(observation, dict):
            serialized = {}
            for k, v in observation.items():
                if isinstance(v, np.ndarray):
                    serialized[k] = {"type": "numpy_array", "data": v.tolist(), "shape": list(v.shape)}
                else:
                    serialized[k] = self._serialize_observation(v)
            return serialized
        elif isinstance(observation, (list, tuple)):
            return [self._serialize_observation(item) for item in observation]
        elif isinstance(observation, np.integer):
            return int(observation)
        elif isinstance(observation, np.floating):
            return float(observation)
        else:
            return observation
    
    def record_multiple_episodes(self, strategies: List[ActionStrategy], 
                                episodes_per_strategy: int = 5, 
                                max_steps_per_episode: int = 1000) -> List[Trajectory]:
        """
        Record multiple episodes using different strategies
        
        Args:
            strategies: List of action strategies to use
            episodes_per_strategy: Number of episodes to record per strategy
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            List of all recorded trajectories
        """
        all_trajectories = []
        
        for strategy in strategies:
            logger.info(f"Recording {episodes_per_strategy} episodes with {strategy.name} strategy")
            
            for episode_num in range(episodes_per_strategy):
                trajectory = self.record_episode(
                    strategy=strategy,
                    max_steps=max_steps_per_episode,
                    episode_id=f"{strategy.name}_ep{episode_num:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                all_trajectories.append(trajectory)
        
        return all_trajectories
    
    def save_trajectories(self, trajectories: List[Trajectory], format: str = "json"):
        """
        Save trajectories to disk
        
        Args:
            trajectories: List of trajectories to save
            format: Save format ("json" or "pickle")
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == "json":
            filename = f"trajectories_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Convert trajectories to dictionaries for JSON serialization
            data = {
                "metadata": {
                    "num_trajectories": len(trajectories),
                    "save_time": timestamp,
                    "format": "json"
                },
                "trajectories": [asdict(traj) for traj in trajectories]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif format == "pickle":
            filename = f"trajectories_{timestamp}.pkl"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(trajectories, f)
        
        logger.info(f"Saved {len(trajectories)} trajectories to {filepath}")
        return filepath
    
    def analyze_trajectories(self, trajectories: List[Trajectory]) -> Dict:
        """
        Analyze recorded trajectories and generate statistics
        
        Args:
            trajectories: List of trajectories to analyze
            
        Returns:
            Analysis results dictionary
        """
        if not trajectories:
            return {}
        
        # Group trajectories by strategy
        strategy_groups = {}
        for traj in trajectories:
            if traj.strategy_name not in strategy_groups:
                strategy_groups[traj.strategy_name] = []
            strategy_groups[traj.strategy_name].append(traj)
        
        analysis = {
            "total_trajectories": len(trajectories),
            "strategies": list(strategy_groups.keys()),
            "strategy_analysis": {}
        }
        
        for strategy_name, trajs in strategy_groups.items():
            rewards = [traj.total_reward for traj in trajs]
            lengths = [traj.episode_length for traj in trajs]
            durations = [traj.end_time - traj.start_time for traj in trajs]
            
            # Action frequency analysis
            action_counts = {}
            for traj in trajs:
                for step in traj.steps:
                    action_name = step.action_name
                    action_counts[action_name] = action_counts.get(action_name, 0) + 1
            
            analysis["strategy_analysis"][strategy_name] = {
                "num_episodes": len(trajs),
                "rewards": {
                    "mean": np.mean(rewards),
                    "std": np.std(rewards),
                    "min": np.min(rewards),
                    "max": np.max(rewards)
                },
                "episode_lengths": {
                    "mean": np.mean(lengths),
                    "std": np.std(lengths),
                    "min": np.min(lengths),
                    "max": np.max(lengths)
                },
                "durations": {
                    "mean": np.mean(durations),
                    "std": np.std(durations),
                    "min": np.min(durations),
                    "max": np.max(durations)
                },
                "action_frequencies": action_counts
            }
        
        return analysis
    
    def close(self):
        """Close the environment"""
        if self.env is not None:
            self.env.close()

def create_strategies(seed: int = 42) -> List[ActionStrategy]:
    """Create a list of different action strategies for testing"""
    strategies = [
        RandomStrategy(seed=seed),
        BalatroSpecificStrategy(seed=seed),
        ConservativeStrategy(seed=seed),
        AggressiveStrategy(seed=seed),
        ShopFocusedStrategy(seed=seed),
        BiasedStrategy({0: 3.0, 1: 2.0, 5: 1.5}, seed=seed),  # Favor play cards and discard
        SequentialStrategy([0, 1, 2, 3, 4, 5]),  # Cycle through basic actions
    ]
    return strategies

def main():
    """Main function to run trajectory recording"""
    parser = argparse.ArgumentParser(description="Record Balatro Gym trajectories")
    parser.add_argument("--env-id", default="Balatro-v0", help="Gym environment ID")
    parser.add_argument("--output-dir", default="trajectories", help="Output directory")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per strategy")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--format", choices=["json", "pickle"], default="json", help="Save format")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create recorder
    recorder = TrajectoryRecorder(env_id=args.env_id, output_dir=args.output_dir)
    
    try:
        # Setup environment
        render_mode = "human" if args.render else None
        recorder.setup_environment(render_mode=render_mode)
        
        # Create strategies
        strategies = create_strategies(seed=args.seed)
        
        # Record trajectories
        logger.info(f"Recording trajectories with {len(strategies)} strategies")
        trajectories = recorder.record_multiple_episodes(
            strategies=strategies,
            episodes_per_strategy=args.episodes,
            max_steps_per_episode=args.max_steps
        )
        
        # Save trajectories
        filepath = recorder.save_trajectories(trajectories, format=args.format)
        
        # Analyze trajectories
        analysis = recorder.analyze_trajectories(trajectories)
        
        # Save analysis
        analysis_file = os.path.join(args.output_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Analysis saved to {analysis_file}")
        
        # Print summary with Balatro-specific details
        print("\n=== BALATRO TRAJECTORY RECORDING SUMMARY ===")
        print(f"Total trajectories recorded: {len(trajectories)}")
        print(f"Strategies used: {', '.join(analysis['strategies'])}")
        print(f"Data saved to: {filepath}")
        print(f"Analysis saved to: {analysis_file}")
        
        for strategy_name, stats in analysis["strategy_analysis"].items():
            print(f"\n{strategy_name.upper()} STRATEGY:")
            print(f"  Episodes: {stats['num_episodes']}")
            print(f"  Avg Reward: {stats['rewards']['mean']:.2f} ± {stats['rewards']['std']:.2f}")
            print(f"  Avg Length: {stats['episode_lengths']['mean']:.1f} steps")
            
            # Show most common actions with Balatro context
            top_actions = sorted(stats['action_frequencies'].items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top actions: {top_actions}")
            
            # Balatro-specific insights
            shop_actions = [action for action, count in top_actions if 'BUY' in action or 'SHOP' in action]
            card_actions = [action for action, count in top_actions if 'PLAY_CARD' in action]
            if shop_actions:
                print(f"  Shop activity: {shop_actions}")
            if card_actions:
                print(f"  Card play activity: {card_actions}")
    
    except KeyboardInterrupt:
        logger.info("Recording interrupted by user")
    except Exception as e:
        logger.error(f"Error during recording: {e}")
        raise
    finally:
        recorder.close()

if __name__ == "__main__":
    main()
