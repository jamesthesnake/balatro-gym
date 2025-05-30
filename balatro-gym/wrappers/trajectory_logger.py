import json
from pathlib import Path

class TrajectoryLogger:
    def __init__(self, save_dir="trajectories"):
        self.path = Path(save_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.steps = []

    def log_step(self, obs, action, reward, done, info):
        self.steps.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "done": done,
            "info": info,
        })

    def save_episode(self, episode_id):
        with open(self.path / f"episode_{episode_id}.jsonl", "w") as f:
            for step in self.steps:
                f.write(json.dumps(step) + "\n")
        self.steps = []
