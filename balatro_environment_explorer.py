#!/usr/bin/env python3
"""
Deep exploration of the Balatro environment to understand its reward structure
"""

import gymnasium as gym
import numpy as np
import json
from pathlib import Path

def explore_balatro_environment():
    """Thoroughly explore the Balatro environment"""
    
    try:
        import balatro_gym
        env = gym.make('Balatro-v0')
        print("✅ Balatro environment loaded")
    except Exception as e:
        print(f"❌ Failed to load Balatro environment: {e}")
        return None
    
    # 1. Environment introspection
    print("\n🔍 ENVIRONMENT INTROSPECTION:")
    print(f"Action space: {env.action_space}")
    print(f"Action space type: {type(env.action_space)}")
    
    if hasattr(env.action_space, 'n'):
        print(f"Number of actions: {env.action_space.n}")
    
    if hasattr(env, 'observation_space'):
        print(f"Observation space: {env.observation_space}")
    
    # Check for environment-specific methods
    env_methods = [method for method in dir(env) if not method.startswith('_')]
    print(f"Environment methods: {env_methods}")
    
    # 2. Initial state exploration
    print("\n📊 INITIAL STATE EXPLORATION:")
    state, info = env.reset()
    
    print(f"State type: {type(state)}")
    print(f"State keys: {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}")
    print(f"Info type: {type(info)}")
    print(f"Info keys: {list(info.keys()) if isinstance(info, dict) else 'Not a dict'}")
    
    # Print actual state values
    if isinstance(state, dict):
        for key, value in state.items():
            if isinstance(value, (list, dict)) and len(str(value)) > 100:
                print(f"  {key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
            else:
                print(f"  {key}: {value}")
    
    if isinstance(info, dict):
        print("\nInfo contents:")
        for key, value in info.items():
            if isinstance(value, (list, dict)) and len(str(value)) > 100:
                print(f"  {key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
            else:
                print(f"  {key}: {value}")
    
    # 3. Action space exploration
    print(f"\n🎮 ACTION SPACE EXPLORATION:")
    
    # Test systematic action sequences
    test_sequences = [
        # Basic actions
        [0, 1, 2, 3, 4],
        # Card selection pattern (if applicable)
        [5, 6, 7, 8, 9],
        # Mixed pattern
        [0, 5, 1, 6, 2],
        # Higher numbers
        [10, 11, 12, 13, 14] if hasattr(env.action_space, 'n') and env.action_space.n > 10 else [0, 1, 2, 3, 4]
    ]
    
    for seq_idx, sequence in enumerate(test_sequences):
        print(f"\n--- Testing sequence {seq_idx + 1}: {sequence} ---")
        
        # Reset for each sequence
        state, info = env.reset()
        total_reward = 0
        
        for step, action in enumerate(sequence):
            # Check if action is valid
            if hasattr(env.action_space, 'contains') and not env.action_space.contains(action):
                print(f"  Step {step + 1}: Action {action} invalid - skipping")
                continue
            
            try:
                result = env.step(action)
                
                # Handle different return formats
                if len(result) == 5:
                    next_state, reward, done, truncated, next_info = result
                else:
                    next_state, reward, done, next_info = result
                    truncated = False
                
                total_reward += reward
                
                print(f"  Step {step + 1}: Action {action} -> Reward {reward}")
                
                # Check for state changes
                if isinstance(state, dict) and isinstance(next_state, dict):
                    changes = {}
                    for key in state.keys():
                        if key in next_state and state[key] != next_state[key]:
                            # Only show scalar changes to avoid clutter
                            if isinstance(state[key], (int, float, str, bool)):
                                changes[key] = f"{state[key]} -> {next_state[key]}"
                    
                    if changes:
                        print(f"    State changes: {changes}")
                
                if reward != 0:
                    print(f"    🎯 NON-ZERO REWARD FOUND! {reward}")
                    print(f"    Current state summary:")
                    if isinstance(next_state, dict):
                        for key, value in next_state.items():
                            if isinstance(value, (int, float, str, bool)):
                                print(f"      {key}: {value}")
                
                if done or truncated:
                    print(f"    Episode ended: done={done}, truncated={truncated}")
                    break
                
                state = next_state
                info = next_info
                
            except Exception as e:
                print(f"  Step {step + 1}: Action {action} -> ERROR: {e}")
                break
        
        print(f"  Sequence total reward: {total_reward}")
    
    # 4. Look for environment documentation or help
    print(f"\n📚 ENVIRONMENT DOCUMENTATION:")
    
    if hasattr(env, '__doc__') and env.__doc__:
        print(f"Environment docstring: {env.__doc__}")
    
    # Check if there are any special methods or attributes
    special_attrs = [attr for attr in dir(env) if 'action' in attr.lower() or 'reward' in attr.lower() or 'info' in attr.lower()]
    if special_attrs:
        print(f"Special attributes: {special_attrs}")
    
    # 5. Try to find reward triggers
    print(f"\n🎯 REWARD TRIGGER EXPLORATION:")
    
    # Test longer sequences to see if rewards come later
    state, info = env.reset()
    
    print("Testing longer random sequence...")
    total_reward = 0
    action_reward_pairs = []
    
    for i in range(50):  # Longer sequence
        # Use a mix of low and higher actions
        if i < 10:
            action = i % 5  # Cycle through first 5 actions
        elif i < 20:
            action = 5 + (i % 5)  # Try actions 5-9
        else:
            action = np.random.randint(0, min(15, env.action_space.n if hasattr(env.action_space, 'n') else 10))
        
        try:
            result = env.step(action)
            
            if len(result) == 5:
                next_state, reward, done, truncated, next_info = result
            else:
                next_state, reward, done, next_info = result
                truncated = False
            
            total_reward += reward
            action_reward_pairs.append((action, reward))
            
            if reward != 0:
                print(f"  Step {i + 1}: Action {action} -> Reward {reward} 🎯")
            elif i % 10 == 0:
                print(f"  Step {i + 1}: Action {action} -> Reward {reward}")
            
            if done or truncated:
                print(f"Episode ended at step {i + 1}")
                break
                
            state = next_state
            
        except Exception as e:
            print(f"  Step {i + 1}: Error with action {action}: {e}")
            break
    
    print(f"Final total reward: {total_reward}")
    
    # Show summary of all rewards
    non_zero_rewards = [(a, r) for a, r in action_reward_pairs if r != 0]
    if non_zero_rewards:
        print(f"All non-zero rewards: {non_zero_rewards}")
    else:
        print("No non-zero rewards found in extended sequence")
    
    env.close()
    
    return {
        'total_reward': total_reward,
        'non_zero_rewards': non_zero_rewards,
        'action_reward_pairs': action_reward_pairs
    }

def check_balatro_gym_source():
    """Try to get insights from the balatro_gym source if available"""
    print("\n🔍 CHECKING BALATRO-GYM SOURCE:")
    
    try:
        import balatro_gym
        
        # Check module location
        print(f"balatro_gym location: {balatro_gym.__file__ if hasattr(balatro_gym, '__file__') else 'Unknown'}")
        
        # Check module contents
        module_contents = [item for item in dir(balatro_gym) if not item.startswith('_')]
        print(f"balatro_gym contents: {module_contents}")
        
        # Look for version or documentation
        if hasattr(balatro_gym, '__version__'):
            print(f"balatro_gym version: {balatro_gym.__version__}")
        
        if hasattr(balatro_gym, '__doc__') and balatro_gym.__doc__:
            print(f"balatro_gym docstring: {balatro_gym.__doc__}")
        
        # Try to find the environment class
        for item_name in module_contents:
            item = getattr(balatro_gym, item_name)
            if hasattr(item, '__doc__') and item.__doc__:
                print(f"{item_name} docstring: {item.__doc__}")
        
    except Exception as e:
        print(f"Could not introspect balatro_gym: {e}")

def create_exploration_report(results):
    """Create a comprehensive report"""
    print("\n" + "="*60)
    print("🎯 BALATRO ENVIRONMENT EXPLORATION REPORT")
    print("="*60)
    
    if results:
        print(f"✅ Environment accessible: Yes")
        print(f"📊 Total reward from exploration: {results['total_reward']}")
        print(f"🎯 Non-zero rewards found: {len(results['non_zero_rewards'])}")
        
        if results['non_zero_rewards']:
            print(f"🏆 Reward-generating actions: {results['non_zero_rewards']}")
            
            # Analyze patterns
            reward_actions = [action for action, reward in results['non_zero_rewards']]
            print(f"📈 Actions that generate rewards: {sorted(set(reward_actions))}")
        else:
            print("❌ No reward-generating actions found")
            print("\n🔧 Possible reasons:")
            print("  1. Environment requires specific game progression")
            print("  2. Rewards only come from winning hands/rounds")
            print("  3. Environment needs proper initialization")
            print("  4. Action space usage is incorrect")
            print("  5. Environment is designed for cumulative rewards")
    else:
        print("❌ Environment not accessible")
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. Check balatro-gym documentation or GitHub repository")
    print("2. Look for example usage in the repository")
    print("3. Try environment with render_mode to see visual feedback")
    print("4. Check if environment has specific initialization parameters")
    print("5. Consider using a hybrid approach with your improved dummy environment")

if __name__ == "__main__":
    print("🎮 Balatro Environment Deep Explorer")
    print("="*50)
    
    # Check source code insights
    check_balatro_gym_source()
    
    # Deep exploration
    results = explore_balatro_environment()
    
    # Generate report
    create_exploration_report(results)
