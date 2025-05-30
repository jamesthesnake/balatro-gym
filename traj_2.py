import multiprocessing as mp
import gymnasium as gym
import numpy as np
import pickle
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import argparse
from tqdm import tqdm
import logging

# Import shared data structures
from data_structures import (
    TrajectoryStep, EpisodeMetadata, DatasetMetadata, 
    BalatroActions, StrategyTags, serialize_state, serialize_info,
    compute_episode_statistics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyTagger:
    """Automatically tag gameplay strategies and phases"""
    
    def tag_step(self, state: Dict, action: int, info: Dict) -> List[str]:
        tags = []
        
        # Basic action classification
        action_name = BalatroActions.get_action_name(action)
        tags.append(f"action_{action_name}")
        
        # Game phase detection based on state
        if self._is_ante_start(state, info):
            tags.append(StrategyTags.ANTE_START)
        if self._is_blind_selection(state, info):
            tags.append(StrategyTags.BLIND_SELECTION)  
        if self._is_shop_phase(state, info):
            tags.append(StrategyTags.SHOP_PHASE)
        if self._is_boss_preparation(state, info):
            tags.append(StrategyTags.BOSS_PREP)
            
        # Strategic patterns
        if self._is_hand_building(state, action):
            tags.append(StrategyTags.HAND_BUILDING)
        if self._is_economy_focus(state, action, info):
            tags.append(StrategyTags.ECONOMY_FOCUS)
        if self._is_conservative_play(state, action):
            tags.append(StrategyTags.CONSERVATIVE)
        if self._is_aggressive_play(state, action):
            tags.append(StrategyTags.AGGRESSIVE)
            
        return tags
    
    def _is_ante_start(self, state: Dict, info: Dict) -> bool:
        # Detect if we're at the start of a new ante
        return state.get('round', 0) == 1 and state.get('hand_count', 0) == 0
    
    def _is_blind_selection(self, state: Dict, info: Dict) -> bool:
        # Detect blind selection phase
        return 'blind_options' in state
    
    def _is_shop_phase(self, state: Dict, info: Dict) -> bool:
        # Detect if we're in shop/purchase phase
        return state.get('phase') == 'shop' or 'shop_items' in state
    
    def _is_boss_preparation(self, state: Dict, info: Dict) -> bool:
        # Detect if we're preparing for boss blind
        return state.get('blind_type') == 'boss' or state.get('ante', 0) % 8 == 0
    
    def _is_hand_building(self, state: Dict, action: int) -> bool:
        # Detect hand building actions (selecting cards)
        return action in [BalatroActions.SELECT_CARD_1, BalatroActions.SELECT_CARD_2, 
                         BalatroActions.SELECT_CARD_3, BalatroActions.SELECT_CARD_4,
                         BalatroActions.SELECT_CARD_5, BalatroActions.SELECT_CARD_6,
                         BalatroActions.SELECT_CARD_7, BalatroActions.SELECT_CARD_8]
    
    def _is_economy_focus(self, state: Dict, action: int, info: Dict) -> bool:
        # Detect economy-focused decisions
        money = state.get('money', 0)
        return money < 5 and action == BalatroActions.DISCARD_HAND
    
    def _is_conservative_play(self, state: Dict, action: int) -> bool:
        # Detect conservative play patterns
        chips_needed = state.get('chips_needed', float('inf'))
        current_chips = state.get('current_chips', 0)
        return current_chips >= chips_needed * 1.5 and action == BalatroActions.PLAY_HAND
    
    def _is_aggressive_play(self, state: Dict, action: int) -> bool:
        # Detect aggressive play patterns  
        chips_needed = state.get('chips_needed', float('inf'))
        current_chips = state.get('current_chips', 0)
        return current_chips < chips_needed and action == BalatroActions.PLAY_HAND

class PolicyBase:
    """Base class for different policies"""
    def select_action(self, state, info):
        raise NotImplementedError

class RandomPolicy(PolicyBase):
    """Random action selection"""
    def __init__(self, action_space_size=10):
        self.action_space_size = action_space_size
    
    def select_action(self, state, info):
        return np.random.randint(0, self.action_space_size)

class HeuristicPolicy(PolicyBase):
    """Simple heuristic policy for Balatro"""
    def select_action(self, state, info):
        # Simple heuristic: 
        # - Play hand if we have cards selected
        # - Otherwise select cards or discard
        
        selected_cards = state.get('selected_cards', [])
        money = state.get('money', 0)
        discards_left = state.get('discards', 0)
        
        # If we have selected cards, play them
        if len(selected_cards) > 0:
            return BalatroActions.PLAY_HAND
        
        # If low on discards and money, be more conservative
        if discards_left <= 1 and money < 5:
            return np.random.randint(BalatroActions.SELECT_CARD_1, BalatroActions.SELECT_CARD_5 + 1)
        
        # Random decision between discard and card selection
        if np.random.random() < 0.3 and discards_left > 0:
            return BalatroActions.DISCARD_HAND
        else:
            return np.random.randint(BalatroActions.SELECT_CARD_1, BalatroActions.SELECT_CARD_8 + 1)

def generate_single_episode(args_tuple):
    """Generate a single episode - designed for multiprocessing"""
    worker_id, episode_id, policy_type, max_steps = args_tuple
    
    try:
        # Initialize environment and policy
        env = create_balatro_env()
        
        if policy_type == 'random':
            policy = RandomPolicy()
        elif policy_type == 'heuristic':
            policy = HeuristicPolicy()
        else:
            policy = RandomPolicy()
        
        strategy_tagger = StrategyTagger()
        trajectory = []
        
        # Episode timing
        start_time = time.time()
        
        # Reset environment
        state, info = env.reset()
        episode_rewards = []
        
        for step in range(max_steps):
            # Select action using policy
            action = policy.select_action(state, info)
            
            # Take step in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Tag the strategy
            strategy_tags = strategy_tagger.tag_step(state, action, info)
            
            # Create trajectory step
            traj_step = TrajectoryStep(
                state=serialize_state(state),
                action=action,
                reward=reward,
                next_state=serialize_state(next_state),
                done=done or truncated,
                info=serialize_info(info),
                rtg=0,  # Will be computed later
                strategy_tags=strategy_tags,
                metadata={
                    'worker_id': worker_id,
                    'episode_id': episode_id,
                    'step_id': step,
                    'timestamp': time.time(),
                    'policy_type': policy_type
                }
            )
            
            trajectory.append(traj_step)
            episode_rewards.append(reward)
            
            if done or truncated:
                break
                
            state = next_state
        
        end_time = time.time()
        
        # Compute reward-to-go
        compute_rtg(trajectory, episode_rewards)
        
        # Create episode metadata
        episode_stats = compute_episode_statistics(trajectory)
        metadata = EpisodeMetadata(
            episode_id=episode_id,
            worker_id=worker_id,
            policy_type=policy_type,
            total_steps=len(trajectory),
            total_reward=sum(episode_rewards),
            episode_length=len(trajectory),
            start_timestamp=start_time,
            end_timestamp=end_time,
            final_ante=trajectory[-1].state.get('ante', 0) if trajectory else 0,
            max_chips=max((step.state.get('chips', 0) for step in trajectory), default=0),
            max_money=max((step.state.get('money', 0) for step in trajectory), default=0),
            strategy_tag_counts=episode_stats.get('strategy_tag_counts', {})
        )
        
        return trajectory, metadata
        
    except Exception as e:
        logger.error(f"Error in worker {worker_id}, episode {episode_id}: {e}")
        return None, None

def compute_rtg(trajectory: List[TrajectoryStep], rewards: List[float]):
    """Compute reward-to-go for each step"""
    rtg = 0
    for i in reversed(range(len(trajectory))):
        rtg += rewards[i]
        trajectory[i].rtg = rtg

def create_balatro_env():
    """Create Balatro environment - adapt based on actual balatro-gym interface"""
    # This is a placeholder - you'll need to implement based on the actual gym interface
    # For now, we'll use a dummy environment
    class DummyBalatroEnv:
        def __init__(self):
            self.action_space_size = 10
            self.step_count = 0
            self.max_steps = 100
        
        def reset(self):
            self.step_count = 0
            state = {
                'hand': [{'rank': 'A', 'suit': '♠'}, {'rank': 'K', 'suit': '♥'}],
                'money': 10,
                'chips': 0,
                'mult': 1,
                'ante': 1,
                'discards': 3,
                'selected_cards': []
            }
            info = {'phase': 'play'}
            return state, info
        
        def step(self, action):
            self.step_count += 1
            
            # Dummy next state
            next_state = {
                'hand': [{'rank': 'Q', 'suit': '♦'}, {'rank': 'J', 'suit': '♣'}], 
                'money': 12,
                'chips': 50,
                'mult': 2,
                'ante': 1,
                'discards': 2,
                'selected_cards': [0] if action >= BalatroActions.SELECT_CARD_1 else []
            }
            
            reward = np.random.uniform(-1, 5)  # Random reward
            done = self.step_count >= self.max_steps or np.random.random() < 0.05
            info = {'phase': 'play', 'chips_needed': 100}
            
            return next_state, reward, done, False, info
    
    return DummyBalatroEnv()

class BalatroDataGenerator:
    def __init__(self, num_workers=None, output_dir="./balatro_dataset"):
        self.num_workers = num_workers or mp.cpu_count()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized data generator with {self.num_workers} workers")
        logger.info(f"Output directory: {self.output_dir}")
    
    def generate_dataset(self, total_episodes=1000, policy_type='random', max_steps_per_episode=200):
        """Generate the complete dataset using multiprocessing"""
        
        logger.info(f"Generating {total_episodes} episodes using {self.num_workers} workers")
        
        # Create argument tuples for each episode
        episodes_per_worker = total_episodes // self.num_workers
        remaining_episodes = total_episodes % self.num_workers
        
        args_list = []
        episode_id = 0
        
        for worker_id in range(self.num_workers):
            worker_episodes = episodes_per_worker + (1 if worker_id < remaining_episodes else 0)
            
            for _ in range(worker_episodes):
                args_list.append((worker_id, episode_id, policy_type, max_steps_per_episode))
                episode_id += 1
        
        # Generate episodes in parallel
        total_steps = 0
        total_reward = 0
        successful_episodes = 0
        episode_metadatas = []
        
        with mp.Pool(self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(generate_single_episode, args_list),
                total=len(args_list),
                desc="Generating episodes"
            ))
        
        # Save results and collect statistics
        for i, (trajectory, episode_metadata) in enumerate(results):
            if trajectory is not None and episode_metadata is not None:
                # Save individual episode
                episode_file = self.output_dir / f"episode_{i:06d}.pkl"
                with open(episode_file, 'wb') as f:
                    pickle.dump(trajectory, f)
                
                # Save episode metadata
                metadata_file = self.output_dir / f"episode_{i:06d}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(episode_metadata.to_dict(), f, indent=2)
                
                total_steps += episode_metadata.total_steps
                total_reward += episode_metadata.total_reward
                successful_episodes += 1
                episode_metadatas.append(episode_metadata)
        
        # Save dataset metadata
        dataset_metadata = DatasetMetadata(
            total_episodes=successful_episodes,
            total_steps=total_steps,
            total_reward=total_reward,
            avg_steps_per_episode=total_steps / max(successful_episodes, 1),
            avg_reward_per_episode=total_reward / max(successful_episodes, 1),
            policy_type=policy_type,
            generation_timestamp=time.time(),
            num_workers=self.num_workers,
            generation_config={
                'max_steps_per_episode': max_steps_per_episode,
                'total_requested_episodes': total_episodes,
                'successful_episodes': successful_episodes
            }
        )
        
        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(dataset_metadata.to_dict(), f, indent=2)
        
        # Save consolidated episode metadata
        episodes_metadata_file = self.output_dir / "episodes_metadata.json"
        with open(episodes_metadata_file, 'w') as f:
            json.dump([meta.to_dict() for meta in episode_metadatas], f, indent=2)
        
        logger.info(f"Dataset generation complete!")
        logger.info(f"Generated {successful_episodes} episodes with {total_steps} total steps")
        logger.info(f"Average reward per episode: {total_reward / max(successful_episodes, 1):.2f}")
        
        return dataset_metadata

def main():
    parser = argparse.ArgumentParser(description="Generate Balatro trajectory dataset")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to generate")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--policy", choices=['random', 'heuristic'], default='random', help="Policy type")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--output-dir", type=str, default="./balatro_dataset", help="Output directory")
    
    args = parser.parse_args()
    
    generator = BalatroDataGenerator(
        num_workers=args.workers,
        output_dir=args.output_dir
    )
    
    metadata = generator.generate_dataset(
        total_episodes=args.episodes,
        policy_type=args.policy,
        max_steps_per_episode=args.max_steps
    )
    
    print(f"\nDataset saved to: {args.output_dir}")
    print(f"Episodes generated: {metadata.total_episodes}")
    print(f"Total steps: {metadata.total_steps}")
    print(f"Average reward: {metadata.avg_reward_per_episode:.2f}")

if __name__ == "__main__":
    main()
