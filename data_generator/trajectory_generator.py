import multiprocessing as mp
import gymnasium as gym
import balatro_gym
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pickle
import json
from pathlib import Path

@dataclass
class TrajectoryStep:
    state: Dict[str, Any]
    action: int
    reward: float
    next_state: Dict[str, Any]
    done: bool
    info: Dict[str, Any]
    rtg: float  # Reward-to-go
    strategy_tags: List[str]
    metadata: Dict[str, Any]

class BalatroDataGenerator:
    def __init__(self, num_workers=mp.cpu_count(), episodes_per_worker=1000):
        self.num_workers = num_workers
        self.episodes_per_worker = episodes_per_worker
        self.strategy_tagger = StrategyTagger()
    
    def generate_episode(self, worker_id: int, episode_id: int) -> List[TrajectoryStep]:
        """Generate a single episode with rich annotations"""
        env = gym.make('Balatro-v0')
        trajectory = []
        
        state, info = env.reset()
        episode_rewards = []
        
        while True:
            # Strategy-aware action selection
            action = self.select_action(state, info)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Tag the current game phase/strategy
            strategy_tags = self.strategy_tagger.tag_step(state, action, info)
            
            # Store step with metadata
            step = TrajectoryStep(
                state=self.serialize_state(state),
                action=action,
                reward=reward,
                next_state=self.serialize_state(next_state),
                done=done or truncated,
                info=info,
                rtg=0,  # Will be computed later
                strategy_tags=strategy_tags,
                metadata={
                    'worker_id': worker_id,
                    'episode_id': episode_id,
                    'step_id': len(trajectory),
                    'timestamp': time.time()
                }
            )
            
            trajectory.append(step)
            episode_rewards.append(reward)
            
            if done or truncated:
                break
            
            state = next_state
        

            self.compute_rtg(trajectory, episode_rewards)
        
            return trajectory

        # Compute reward-to-go for each step
