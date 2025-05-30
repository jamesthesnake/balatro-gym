#!/usr/bin/env python3
"""
Debug version of episode generation with detailed reward tracking
"""

import gymnasium as gym
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

def debug_generate_episode(policy_type='heuristic', max_steps=50):
    """Generate a single episode with detailed debugging"""
    
    print(f"🎮 Starting debug episode with {policy_type} policy")
    
    # Create environment
    try:
        import balatro_gym
        env = gym.make('Balatro-v0')
        print("✅ Using real Balatro environment")
        env_type = "real"
    except Exception as e:
        print(f"❌ Failed to load real environment: {e}")
        return None
    
    # Create a simple policy for testing
    def simple_policy(state, info, step_num):
        """Simple policy that tries different action patterns"""
        if step_num < 5:
            return step_num  # Try actions 0, 1, 2, 3, 4
        else:
            return np.random.randint(0, min(10, env.action_space.n))
    
    # Reset and start episode
    state, info = env.reset()
    print(f"📊 Initial state keys: {list(state.keys())}")
    print(f"📋 Initial info: {info}")
    
    episode_rewards = []
    episode_actions = []
    episode_states = []
    
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        
        # Log current state
        state_summary = {}
        for key in ['chips', 'money', 'hands_left', 'discards', 'ante']:
            if key in state:
                state_summary[key] = state[key]
        print(f"State: {state_summary}")
        
        # Select action
        action = simple_policy(state, info, step)
        print(f"Action: {action}")
        episode_actions.append(action)
        episode_states.append(state_summary.copy())
        
        # Take step
        try:
            next_state, reward, done, truncated, next_info = env.step(action)
            
            print(f"Reward: {reward} (type: {type(reward)})")
            print(f"Done: {done}, Truncated: {truncated}")
            
            # Log state changes
            next_state_summary = {}
            for key in ['chips', 'money', 'hands_left', 'discards', 'ante']:
                if key in next_state:
                    next_state_summary[key] = next_state[key]
            
            print(f"Next state: {next_state_summary}")
            
            # Check for state changes
            changes = {}
            for key in state_summary:
                if key in next_state_summary and state_summary[key] != next_state_summary[key]:
                    changes[key] = f"{state_summary[key]} -> {next_state_summary[key]}"
            
            if changes:
                print(f"Changes: {changes}")
            else:
                print("No state changes detected")
            
            episode_rewards.append(reward)
            
            if done or truncated:
                print(f"Episode ended: done={done}, truncated={truncated}")
                break
                
            state = next_state
            info = next_info
            
        except Exception as e:
            print(f"❌ Error on step {step + 1}: {e}")
            break
    
    # Episode summary
    print(f"\n🎯 EPISODE SUMMARY:")
    print(f"Total steps: {len(episode_rewards)}")
    print(f"Total reward: {sum(episode_rewards)}")
    print(f"Rewards: {episode_rewards}")
    print(f"Actions: {episode_actions}")
    print(f"Non-zero rewards: {[r for r in episode_rewards if r != 0]}")
    print(f"Reward statistics:")
    if episode_rewards:
        print(f"  Min: {min(episode_rewards)}")
        print(f"  Max: {max(episode_rewards)}")
        print(f"  Mean: {np.mean(episode_rewards):.4f}")
        print(f"  Non-zero count: {sum(1 for r in episode_rewards if r != 0)}")
    
    env.close()
    
    return {
        'total_reward': sum(episode_rewards),
        'rewards': episode_rewards,
        'actions': episode_actions,
        'states': episode_states,
        'steps': len(episode_rewards)
    }

def test_action_space(max_actions=20):
    """Test different actions to see which ones generate rewards"""
    
    print(f"🔍 Testing action space (up to {max_actions} actions)")
    
    try:
        import balatro_gym
        env = gym.make('Balatro-v0')
        print(f"Action space: {env.action_space}")
        print(f"Action space size: {env.action_space.n}")
    except Exception as e:
        print(f"❌ Failed to load environment: {e}")
        return
    
    # Test each action systematically
    action_rewards = {}
    
    for action in range(min(max_actions, env.action_space.n)):
        print(f"\n🎯 Testing action {action}")
        
        # Reset for each test
        state, info = env.reset()
        
        try:
            next_state, reward, done, truncated, next_info = env.step(action)
            action_rewards[action] = reward
            
            print(f"  Reward: {reward}")
            if reward != 0:
                print(f"  ⭐ NON-ZERO REWARD FOUND!")
                # Log what changed
                state_changes = {}
                for key in ['chips', 'money', 'hands_left', 'discards']:
                    if key in state and key in next_state and state[key] != next_state[key]:
                        state_changes[key] = f"{state[key]} -> {next_state[key]}"
                if state_changes:
                    print(f"  Changes: {state_changes}")
                    
        except Exception as e:
            print(f"  ❌ Error: {e}")
            action_rewards[action] = "ERROR"
    
    env.close()
    
    # Summary
    print(f"\n📊 ACTION REWARD SUMMARY:")
    for action, reward in action_rewards.items():
        if reward != 0 and reward != "ERROR":
            print(f"  Action {action}: {reward} ⭐")
        else:
            print(f"  Action {action}: {reward}")
    
    non_zero_actions = [a for a, r in action_rewards.items() if r != 0 and r != "ERROR"]
    print(f"\n🎯 Actions that give rewards: {non_zero_actions}")
    
    return action_rewards

if __name__ == "__main__":
    print("🐛 Balatro Reward Debugging")
    print("=" * 40)
    
    # Test 1: Action space exploration
    print("🔍 Phase 1: Testing action space")
    action_results = test_action_space()
    
    # Test 2: Full episode
    print("\n🔍 Phase 2: Testing full episode")
    episode_result = debug_generate_episode()
    
    # Recommendations
    print("\n" + "=" * 40)
    print("🎯 DEBUGGING RESULTS:")
    
    if action_results:
        non_zero_actions = [a for a, r in action_results.items() if isinstance(r, (int, float)) and r != 0]
        if non_zero_actions:
            print(f"✅ Found reward-generating actions: {non_zero_actions}")
        else:
            print("❌ No actions generated non-zero rewards")
    
    if episode_result and episode_result['total_reward'] != 0:
        print(f"✅ Episode generated total reward: {episode_result['total_reward']}")
    else:
        print("❌ Episode generated zero total reward")
    
    print("\n🔧 Next steps:")
    print("1. Check Balatro environment documentation for correct action usage")
    print("2. Verify if environment requires specific sequences of actions")
    print("3. Check if rewards are only given at episode completion")
    print("4. Consider if environment needs specific initialization parameters")
