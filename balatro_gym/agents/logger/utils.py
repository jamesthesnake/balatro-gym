import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

class EpisodeLogger:
    """Logs episodes for analysis and debugging."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.log_dir / "episodes").mkdir(exist_ok=True)
        (self.log_dir / "summaries").mkdir(exist_ok=True)
        
        # Initialize CSV for summary stats
        self.summary_file = self.log_dir / "summaries" / f"summary_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self._init_summary_csv()
    
    def _init_summary_csv(self):
        """Initialize the summary CSV file."""
        with open(self.summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'episode', 'agent', 'final_ante', 'final_money', 
                'total_reward', 'termination_reason', 'duration_seconds',
                'total_actions', 'avg_hand_score'
            ])
            writer.writeheader()
    
    def log_episode(self, episode_num: int, agent_name: str, 
                    env_summary: Dict[str, Any], total_reward: float,
                    duration: float):
        """Log a complete episode."""
        # Detailed episode log
        episode_data = {
            'episode': episode_num,
            'agent': agent_name,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'total_reward': total_reward,
            'summary': env_summary
        }
        
        episode_file = self.log_dir / "episodes" / f"episode_{episode_num:04d}.json"
        with open(episode_file, 'w') as f:
            json.dump(episode_data, f, indent=2)
        
        # Summary row
        summary_row = {
            'episode': episode_num,
            'agent': agent_name,
            'final_ante': env_summary.get('final_ante', 1),
            'final_money': env_summary.get('final_money', 0),
            'total_reward': total_reward,
            'termination_reason': env_summary.get('termination_reason', 'unknown'),
            'duration_seconds': duration,
            'total_actions': len(env_summary.get('actions', [])),
            'avg_hand_score': env_summary.get('average_hand_score', 0)
        }
        
        with open(self.summary_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_row.keys())
            writer.writerow(summary_row)
    
    def log_best_episode(self, episode_data: Dict[str, Any]):
        """Save the best episode separately for analysis."""
        best_file = self.log_dir / "best_episode.json"
        with open(best_file, 'w') as f:
            json.dump(episode_data, f, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics from logged episodes."""
        import pandas as pd
        
        try:
            df = pd.read_csv(self.summary_file)
            
            stats = {
                'total_episodes': len(df),
                'avg_ante_reached': df['final_ante'].mean(),
                'max_ante_reached': df['final_ante'].max(),
                'win_rate': (df['termination_reason'] == 'victory').mean(),
                'avg_total_reward': df['total_reward'].mean(),
                'avg_duration': df['duration_seconds'].mean(),
            }
            
            # Group by termination reason
            termination_counts = df['termination_reason'].value_counts().to_dict()
            stats['termination_reasons'] = termination_counts
            
            return stats
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return {}
