#!/usr/bin/env python3
"""
Example script showing how to use an LLM agent to play Balatro.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from balatro_gym.env import BalatroEnv
from agents.llm_agent import LLMBalatroAgent
from utils.logger import EpisodeLogger

# Example using OpenAI API
def create_openai_client():
    """Create OpenAI client - replace with your LLM of choice."""
    import openai
    import os
    
    # Make sure to set OPENAI_API_KEY environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return openai

def main():
    # Initialize environment
    env = BalatroEnv(render_mode="human")
    
    # Initialize LLM client (adapt to your LLM)
    llm_client = create_openai_client()
    
    # Create agent
    agent = LLMBalatroAgent(
        llm_client=llm_client,
        name="GPT-4-Balatro",
        temperature=0.7,
        use_cot=True  # Use chain-of-thought reasoning
    )
    
    # Initialize logger
    logger = EpisodeLogger()
    
    # Run episodes
    num_episodes = 10
    best_reward = float('-inf')
    best_episode = None
    
    for episode in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*50}\n")
        
        # Reset environment
        obs, info = env.reset()
        agent.on_episode_start()
        
        # Track episode
        total_reward = 0.0
        start_time = time.time()
        done = False
        
        while not done:
            # Get action from agent
            action_mask = env.action_mask()
            action = agent.get_action(obs, action_mask, env)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Optional: Add delay for human viewing
            if env.render_mode == "human":
                time.sleep(0.5)
        
        # Episode complete
        duration = time.time() - start_time
        episode_summary = env.get_episode_summary()
        
        print(f"\nEpisode Complete!")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Ante: {episode_summary['final_ante']}")
        print(f"Termination: {episode_summary['termination_reason']}")
        
        # Log episode
        logger.log_episode(
            episode_num=episode + 1,
            agent_name=agent.name,
            env_summary=episode_summary,
            total_reward=total_reward,
            duration=duration
        )
        
        # Track best episode
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = {
                'episode': episode + 1,
                'reward': total_reward,
                'summary': episode_summary
            }
        
        agent.on_episode_end(total_reward, episode_summary)
    
    # Save best episode
    if best_episode:
        logger.log_best_episode(best_episode)
    
    # Print final statistics
    print(f"\n{'='*50}")
    print("Final Statistics")
    print(f"{'='*50}")
    
    stats = logger.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\nBest episode: #{best_episode['episode']} with reward {best_episode['reward']:.2f}")

if __name__ == "__main__":
    main()
