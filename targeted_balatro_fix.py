#!/usr/bin/env python3
"""
Targeted fix for the Balatro environment issues
"""

import gymnasium as gym
import numpy as np
import logging
from typing import Dict, Any, Optional
import copy

logger = logging.getLogger(__name__)

class FixedBalatroWrapper:
    """Fixed wrapper that handles the array comparison bug and focuses on reward-generating actions"""
    
    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode
        self.env = None
        self.step_count = 0
        self.episode_rewards = []
        self.episode_actions = []
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize with better error handling"""
        try:
            import balatro_gym
            self.env = gym.make('Balatro-v0', render_mode=self.render_mode)
            logger.info("✅ Real Balatro environment initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            raise e
    
    def reset(self, **kwargs):
        """Reset with safe state handling"""
        self.step_count = 0
        self.episode_rewards = []
        self.episode_actions = []
        
        state, info = self.env.reset(**kwargs)
        
        # Safely copy state to avoid reference issues
        safe_state = self._make_state_safe(state)
        safe_info = self._make_info_safe(info)
        
        return safe_state, safe_info
    
    def step(self, action):
        """Step with error handling and safe state management"""
        self.step_count += 1
        
        try:
            result = self.env.step(action)
            
            # Handle different return formats
            if len(result) == 5:
                next_state, reward, done, truncated, info = result
            else:
                next_state, reward, done, info = result
                truncated = False
            
            # Safely process state
            safe_next_state = self._make_state_safe(next_state)
            safe_info = self._make_info_safe(info)
            
            # Track episode data
            self.episode_rewards.append(reward)
            self.episode_actions.append(action)
            
            # Log significant events
            if reward != 0:
                logger.info(f"🎯 Reward at step {self.step_count}: {reward}")
            
            return safe_next_state, reward, done, truncated, safe_info
            
        except Exception as e:
            logger.error(f"❌ Error in step {self.step_count} with action {action}: {e}")
            # Return a safe failure state instead of crashing
            return {}, -1.0, True, True, {}
    
    def _make_state_safe(self, state):
        """Convert state to safe format that won't cause array comparison issues"""
        if not isinstance(state, dict):
            return state
        
        safe_state = {}
        for key, value in state.items():
            try:
                if isinstance(value, np.ndarray):
                    # Convert numpy arrays to lists to avoid comparison issues
                    safe_state[key] = value.tolist()
                elif isinstance(value, list):
                    # Deep copy lists to avoid reference issues
                    safe_state[key] = copy.deepcopy(value)
                else:
                    safe_state[key] = value
            except Exception as e:
                logger.warning(f"Could not process state key {key}: {e}")
                safe_state[key] = str(value)  # Fallback to string representation
        
        return safe_state
    
    def _make_info_safe(self, info):
        """Convert info to safe format"""
        if not isinstance(info, dict):
            return info
        
        safe_info = {}
        for key, value in info.items():
            try:
                if isinstance(value, np.ndarray):
                    safe_info[key] = value.tolist()
                elif isinstance(value, list):
                    safe_info[key] = copy.deepcopy(value)
                else:
                    safe_info[key] = value
            except Exception as e:
                logger.warning(f"Could not process info key {key}: {e}")
                safe_info[key] = str(value)
        
        return safe_info
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

class RewardFocusedPolicy:
    """Policy that focuses on actions that generated rewards in exploration"""
    
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        # Based on exploration: action 5 gave reward 0.05
        self.reward_actions = [5, 6, 7, 8, 9]  # Actions 5-9 seem promising
        self.safe_actions = [0, 1, 2, 3, 4]    # Basic actions
        self.step_count = 0
    
    def select_action(self, state, info):
        """Select actions that are more likely to generate rewards"""
        self.step_count += 1
        
        # Start with reward-generating actions
        if self.step_count <= 10:
            # Try reward-generating actions first
            return self.rng.choice(self.reward_actions)
        elif self.step_count <= 20:
            # Mix of reward and safe actions
            all_actions = self.reward_actions + self.safe_actions
            return self.rng.choice(all_actions)
        else:
            # Random from known safe range
            return self.rng.choice(self.safe_actions)

def test_fixed_environment():
    """Test the fixed environment"""
    print("🧪 Testing Fixed Balatro Environment")
    
    env = FixedBalatroWrapper()
    policy = RewardFocusedPolicy()
    
    # Test episode
    state, info = env.reset()
    print(f"✅ Environment reset successfully")
    print(f"📊 Initial state keys: {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}")
    
    total_reward = 0
    successful_steps = 0
    
    for step in range(30):
        action = policy.select_action(state, info)
        
        try:
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            successful_steps += 1
            
            if reward != 0:
                print(f"🎯 Step {step + 1}: Action {action} -> Reward {reward}")
            elif step % 5 == 0:
                print(f"📍 Step {step + 1}: Action {action} -> Reward {reward}")
            
            if done or truncated:
                print(f"Episode ended at step {step + 1}: done={done}, truncated={truncated}")
                break
            
            state = next_state
            
        except Exception as e:
            print(f"❌ Error at step {step + 1}: {e}")
            break
    
    print(f"\n📊 Test Results:")
    print(f"  Successful steps: {successful_steps}")
    print(f"  Total reward: {total_reward}")
    print(f"  Episode rewards: {env.episode_rewards}")
    print(f"  Episode actions: {env.episode_actions}")
    
    env.close()
    
    return {
        'successful_steps': successful_steps,
        'total_reward': total_reward,
        'rewards': env.episode_rewards,
        'actions': env.episode_actions
    }

def create_fixed_episode_generator():
    """Create a fixed version of your episode generator"""
    
    code = '''
def generate_fixed_episode(args_tuple):
    """Fixed episode generation that handles the environment properly"""
    worker_id, episode_id, policy_type, max_steps = args_tuple
    
    try:
        # Use fixed wrapper
        env_wrapper = FixedBalatroWrapper()
        
        # Use reward-focused policy or your existing policies
        if policy_type == 'reward_focused':
            policy = RewardFocusedPolicy()
        elif policy_type == 'heuristic':
            policy = HeuristicPolicy(aggression=0.5)
        else:
            policy = RewardFocusedPolicy()  # Default to reward-focused
        
        trajectory = []
        
        # Reset environment
        state, info = env_wrapper.reset()
        episode_rewards = []
        
        for step in range(max_steps):
            # Select action
            action = policy.select_action(state, info)
            
            # Take step with error handling
            try:
                next_state, reward, done, truncated, info = env_wrapper.step(action)
                
                # Create trajectory step (your existing code)
                traj_step = TrajectoryStep(
                    state=state,  # Already safe from wrapper
                    action=action,
                    reward=reward,
                    next_state=next_state,  # Already safe from wrapper
                    done=done or truncated,
                    info=info,  # Already safe from wrapper
                    rtg=0,  # Will be computed later
                    strategy_tags=[],  # Add your strategy tagging
                    metadata={
                        'worker_id': worker_id,
                        'episode_id': episode_id,
                        'step_id': step,
                        'timestamp': time.time(),
                        'policy_type': policy_type,
                        'env_type': 'fixed_balatro'
                    }
                )
                
                trajectory.append(traj_step)
                episode_rewards.append(reward)
                
                if done or truncated:
                    break
                    
                state = next_state
                
            except Exception as e:
                logger.warning(f"Step {step} failed in episode {episode_id}: {e}")
                break
        
        # Clean up
        env_wrapper.close()
        
        # Compute RTG (your existing code)
        compute_rtg(trajectory, episode_rewards)
        
        # Create metadata
        metadata = {
            'episode_id': episode_id,
            'worker_id': worker_id,
            'policy_type': policy_type,
            'total_steps': len(trajectory),
            'total_reward': sum(episode_rewards),
            'successful_steps': len(trajectory),
            'env_type': 'fixed_balatro'
        }
        
        return trajectory, metadata
        
    except Exception as e:
        logger.error(f"Error in worker {worker_id}, episode {episode_id}: {e}")
        return None, None
'''

    print("🔧 FIXED INTEGRATION:")
    print("Replace your generate_real_episode function with:")
    print(code)

if __name__ == "__main__":
    print("🎯 Fixed Balatro Environment Test")
    print("="*40)
    
    # Test the fix
    test_results = test_fixed_environment()
    
    if test_results['total_reward'] > 0:
        print("✅ SUCCESS! Fixed environment is generating rewards!")
        print(f"🎯 Generated {test_results['total_reward']} total reward")
        print(f"📈 Successful steps: {test_results['successful_steps']}")
        
        create_fixed_episode_generator()
        
        print("\n💡 Next steps:")
        print("1. Replace your BalatroEnvironmentWrapper with FixedBalatroWrapper")
        print("2. Use RewardFocusedPolicy as default policy")
        print("3. Re-run your dataset generation")
    else:
        print("❌ Still having issues - may need deeper investigation")
        print("Consider checking the balatro-gym repository for documentation")
