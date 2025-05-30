import json
from pathlib import Path
import numpy as np # Make sure to import numpy

class TrajectoryLogger:
    def __init__(self, save_dir="trajectories"):
        self.path = Path(save_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.steps = []

    def log_step(self, obs, action, reward, done, info): # Assuming 'done' is what your script passes
        self.steps.append({
            "obs": obs,
            "action": action, # Will be handled by _to_json_serializable if it's an Enum
            "reward": reward,
            "done": done,
            "info": info,
        })

    def _to_json_serializable(self, data):
        """
        Recursively converts data to a JSON-serializable format.
        Handles numpy arrays, numpy numeric types, and simple enums.
        """
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.int_)): # Handles np.int8, np.int32, np.int64 etc.
            return int(data)
        elif isinstance(data, (np.floating, np.float_)): # Handles np.float32, np.float64 etc.
            return float(data)
        elif isinstance(data, np.bool_): # Handles np.bool_
            return bool(data)
        elif isinstance(data, dict):
            return {k: self._to_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._to_json_serializable(item) for item in data]
        # Handling for Enums (like your Action enum if it's passed directly)
        # Your Action enum is an IntEnum, json.dumps might handle its .value,
        # but this makes it explicit.
        elif hasattr(data, 'value') and isinstance(data.value, (int, float, str, bool)):
            return data.value
        return data # Return data as is if it's already serializable or not a handled type

    def save_episode(self, episode_id):
        filepath = self.path / f"episode_{episode_id}.jsonl"
        try:
            with open(filepath, "w") as f:
                for step_data in self.steps:
                    # Convert the step data to be JSON serializable
                    serializable_step = self._to_json_serializable(step_data)
                    f.write(json.dumps(serializable_step) + "\n")
            # print(f"Successfully saved episode {episode_id} to {filepath}") # Optional: for confirmation
        except Exception as e:
            print(f"Error saving episode {episode_id} to {filepath}: {e}")
            # You might want to log the problematic step_data or re-raise the exception
            # For debugging, you could try to serialize parts of serializable_step:
            # for key, value in serializable_step.items():
            #     try:
            #         json.dumps({key: value})
            #     except TypeError:
            #         print(f"Problematic key: {key}, value type: {type(value)}, value: {value}")
            raise # Re-raise the exception if you want the script to stop on error

        self.steps = [] # Clear buffer for the next episode
