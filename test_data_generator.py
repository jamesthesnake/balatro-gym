#!/usr/bin/env python3
"""
Simple test data generator to create sample Balatro dataset for testing.
Run this first to generate test data, then use the Streamlit viewer.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import time

# Define TrajectoryStep here to avoid import issues during testing
@dataclass
class TrajectoryStep:
    state: Dict[str, Any]
    action: int
    reward: float
    next_state: Dict[str, Any]
    done: bool
    info: Dict[str, Any]
    rtg: float
    strategy_tags: List[str]
    metadata: Dict[str, Any]

def generate_test_episode(episode_id: int, num_steps: int = 50) -> List[TrajectoryStep]:
    """Generate a single test episode with realistic-looking data."""
    
    trajectory = []
    total_reward = 0
    
    # Starting game state
    current_state = {
        'hand': [
            {'rank': 'A', 'suit': '♠'}, {'rank': 'K', 'suit': '♥'},
            {'rank': 'Q', 'suit': '♦'}, {'rank': 'J', 'suit': '♣'},
            {'rank': '10', 'suit': '♠'}
        ],
        'money': 10,
        'chips': 0,
        'mult': 1,
        'ante': 1,
        'discards': 3,
        'hands_left': 4,
        'selected_cards': []
    }
    
    for step_id in range(num_steps):
        # Random action (0-9 based on balatro-gym)
        action = np.random.randint(0, 10)
        
        # Random reward with some structure
        if action == 0:  # Play hand
            reward = np.random.uniform(10, 100)
        elif action == 1:  # Discard
            reward = np.random.uniform(-5, 5)
        else:  # Select cards
            reward = np.random.uniform(-1, 1)
        
        total_reward += reward
        
        # Next state (slightly modified)
        next_state = current_state.copy()
        next_state['chips'] += max(0, reward)
        next_state['money'] += np.random.randint(-2, 3)
        next_state['money'] = max(0, next_state['money'])
        
        # Strategy tags based on action
        strategy_tags = []
        if action == 0:
            strategy_tags = ['action_play_hand', 'aggressive']
        elif action == 1:
            strategy_tags = ['action_discard_hand', 'conservative']
        else:
            strategy_tags = ['action_select_card', 'hand_building']
        
        # Add some random strategy tags
        if np.random.random() < 0.3:
            strategy_tags.append('economy_focus')
        if step_id < 5:
            strategy_tags.append('ante_start')
        
        # Determine if episode should end
        done = step_id >= num_steps - 1 or np.random.random() < 0.02
        
        # Create trajectory step
        step = TrajectoryStep(
            state=current_state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            info={'phase': 'play', 'chips_needed': 300},
            rtg=0,  # Will be computed later
            strategy_tags=strategy_tags,
            metadata={
                'episode_id': episode_id,
                'step_id': step_id,
                'timestamp': time.time(),
                'policy_type': 'test'
            }
        )
        
        trajectory.append(step)
        current_state = next_state
        
        if done:
            break
    
    # Compute reward-to-go
    rtg = 0
    for i in reversed(range(len(trajectory))):
        rtg += trajectory[i].reward
        trajectory[i].rtg = rtg
    
    return trajectory

def generate_test_dataset(num_episodes: int = 20, output_dir: str = "./test_balatro_dataset"):
    """Generate a test dataset with multiple episodes."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Generating {num_episodes} test episodes...")
    
    total_steps = 0
    total_reward = 0
    
    for episode_id in range(num_episodes):
        # Generate episode
        episode = generate_test_episode(episode_id, num_steps=np.random.randint(20, 80))
        
        # Save episode
        episode_file = output_path / f"episode_{episode_id:06d}.pkl"
        with open(episode_file, 'wb') as f:
            pickle.dump(episode, f)
        
        # Update statistics
        total_steps += len(episode)
        total_reward += sum(step.reward for step in episode)
        
        print(f"Generated episode {episode_id}: {len(episode)} steps, {sum(step.reward for step in episode):.1f} total reward")
    
    # Create dataset metadata
    metadata = {
        'total_episodes': num_episodes,
        'total_steps': total_steps,
        'total_reward': total_reward,
        'avg_steps_per_episode': total_steps / num_episodes,
        'avg_reward_per_episode': total_reward / num_episodes,
        'policy_type': 'test',
        'generation_timestamp': time.time(),
        'num_workers': 1,
        'dataset_version': '1.0-test'
    }
    
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTest dataset generated successfully!")
    print(f"Location: {output_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Average reward per episode: {total_reward / num_episodes:.2f}")
    print(f"\nTo view the dataset, run:")
    print(f"streamlit run streamlit_viewer.py")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test Balatro dataset")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to generate")
    parser.add_argument("--output-dir", type=str, default="./test_balatro_dataset", help="Output directory")
    
    args = parser.parse_args()
    
    generate_test_dataset(args.episodes, args.output_dir)
