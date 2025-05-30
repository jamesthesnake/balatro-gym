#!/usr/bin/env python3
"""
Debug script to understand why Balatro rewards are zero
"""

import os
import sys
import gymnasium as gym
import numpy as np
import json
import pickle
from pathlib import Path

def test_real_balatro_environment():
    """Test the real Balatro environment to understand reward structure"""
    print("🔍 Testing Real Balatro Environment...")
    
    try:
        # Try to import and create real environment
        import balatro_gym
        env = gym.make('Balatro-v0')
        print("✅ Real Balatro environment loaded successfully")
        
        # Reset and get initial state
        state, info = env.reset()
        print(f"📊 Initial state keys: {list(state.keys())}")
        print(f"📋 Initial info keys: {list(info.keys())}")
        
        # Print some initial values
        if 'chips' in state:
            print(f"💰 Initial chips: {state['chips']}")
        if 'money' in state:
            print(f"💵 Initial money: {state['money']}")
        if 'hands_left' in state:
            print(f"🃏 Hands left: {state['hands_left']}")
            
        print(f"ℹ️ Info: {info}")
        
        # Test several actions and track rewards
        print("\n🎮 Testing actions and rewards:")
        total_reward = 0
        
        for i in range(10):
            # Try different actions
            if i < 5:
                action = i % 5  # First 5 basic actions
            else:
                action = np.random.randint(0, env.action_space.n)
            
            print(f"\n--- Step {i+1} ---")
            print(f"Action: {action}")
            print(f"State before: chips={state.get('chips', 0)}, money={state.get('money', 0)}")
            
            try:
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                print(f"Reward: {reward}")
                print(f"State after: chips={next_state.get('chips', 0)}, money={next_state.get('money', 0)}")
                print(f"Done: {done}, Truncated: {truncated}")
                print(f"Info: {info}")
                
                state = next_state
                
                if done or truncated:
                    print(f"Episode ended at step {i+1}")
                    break
                    
            except Exception as e:
                print(f"❌ Error on step {i+1}: {e}")
                break
        
        print(f"\n📈 Total reward collected: {total_reward}")
        env.close()
        
        return total_reward != 0
        
    except ImportError:
        print("❌ balatro_gym not available")
        return False
    except Exception as e:
        print(f"❌ Error testing real environment: {e}")
        return False

def analyze_generated_dataset(dataset_path="./realistic_balatro_dataset"):
    """Analyze the generated dataset to understand reward distribution"""
    print(f"\n🔍 Analyzing generated dataset at: {dataset_path}")
    
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        print(f"❌ Dataset directory doesn't exist: {dataset_path}")
        return
    
    # Load dataset metadata
    metadata_file = dataset_dir / "dataset_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"📊 Dataset metadata:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
    
    # Analyze individual episodes
    episode_files = list(dataset_dir.glob("episode_*.pkl"))
    print(f"\n📁 Found {len(episode_files)} episode files")
    
    if len(episode_files) > 0:
        # Analyze first few episodes
        for i, episode_file in enumerate(episode_files[:5]):
            print(f"\n--- Episode {i+1} ({episode_file.name}) ---")
            
            try:
                with open(episode_file, 'rb') as f:
                    trajectory = pickle.load(f)
                
                print(f"Steps in trajectory: {len(trajectory)}")
                
                if trajectory:
                    # Analyze rewards
                    rewards = [step.reward for step in trajectory]
                    total_reward = sum(rewards)
                    
                    print(f"Rewards: {rewards[:10]}..." if len(rewards) > 10 else f"Rewards: {rewards}")
                    print(f"Total reward: {total_reward}")
                    print(f"Min reward: {min(rewards) if rewards else 0}")
                    print(f"Max reward: {max(rewards) if rewards else 0}")
                    print(f"Non-zero rewards: {sum(1 for r in rewards if r != 0)}")
                    
                    # Analyze first few states and actions
                    for j, step in enumerate(trajectory[:3]):
                        print(f"  Step {j+1}: action={step.action}, reward={step.reward}")
                        if hasattr(step, 'state') and step.state:
                            state_summary = {k: v for k, v in step.state.items() if k in ['chips', 'money', 'hands_left', 'discards']}
                            print(f"    State: {state_summary}")
                
            except Exception as e:
                print(f"❌ Error loading episode {episode_file}: {e}")

def test_wrapper_environment():
    """Test the wrapper environment to see if it's using dummy or real"""
    print("\n🔍 Testing BalatroEnvironmentWrapper...")
    
    # Import the wrapper from your code
    sys.path.append('.')
    
    try:
        # We'll recreate the wrapper logic here since we can't import directly
        class TestWrapper:
            def __init__(self):
                self.env = None
                self._initialize_environment()
            
            def _initialize_environment(self):
                try:
                    import balatro_gym
                    self.env = gym.make('Balatro-v0')
                    self.env_type = "real"
                    print("✅ Using real Balatro environment")
                except Exception as e:
                    print(f"⚠️ Real environment failed: {e}")
                    print("🔄 Would fallback to dummy environment")
                    self.env_type = "dummy"
            
            def test_rewards(self):
                if self.env_type == "real":
                    state, info = self.env.reset()
                    total_reward = 0
                    
                    for i in range(5):
                        action = i % self.env.action_space.n
                        next_state, reward, done, truncated, info = self.env.step(action)
                        print(f"Action {action}: reward = {reward}")
                        total_reward += reward
                        state = next_state
                        if done or truncated:
                            break
                    
                    return total_reward
                else:
                    return None
        
        wrapper = TestWrapper()
        reward_total = wrapper.test_rewards()
        
        if reward_total is not None:
            print(f"Total rewards from wrapper test: {reward_total}")
            return reward_total != 0
        else:
            print("Could not test rewards (using dummy environment)")
            return False
            
    except Exception as e:
        print(f"❌ Error testing wrapper: {e}")
        return False

def suggest_fixes():
    """Suggest potential fixes for the reward issue"""
    print("\n🔧 Potential fixes for zero rewards:")
    print("1. Check if the real Balatro environment returns rewards in a different format")
    print("2. Verify that actions are being processed correctly by the environment")
    print("3. Check if rewards are cumulative vs per-step")
    print("4. Look at the Balatro environment documentation for expected action space")
    print("5. Add debugging prints in the episode generation function")
    print("6. Check if the environment needs specific initialization parameters")

def main():
    print("🎯 Balatro Reward Debugging Tool")
    print("=" * 50)
    
    # Test 1: Real environment
    real_env_works = test_real_balatro_environment()
    
    # Test 2: Generated dataset
    analyze_generated_dataset()
    
    # Test 3: Wrapper
    wrapper_works = test_wrapper_environment()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 DEBUGGING SUMMARY:")
    print(f"✅ Real environment gives rewards: {real_env_works}")
    print(f"✅ Wrapper environment works: {wrapper_works}")
    
    if not real_env_works:
        print("\n❌ ISSUE IDENTIFIED: Real Balatro environment returns zero rewards")
        print("This could mean:")
        print("- The environment needs specific actions to generate rewards")
        print("- Rewards are only given at episode end")
        print("- The environment has a different reward structure than expected")
        
    suggest_fixes()

if __name__ == "__main__":
    main()
