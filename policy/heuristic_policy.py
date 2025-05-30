"""
Human Replay Policy for Balatro Dataset Generation

This policy replays recorded human gameplay sequences, allowing the dataset
to include high-quality human demonstrations alongside algorithmic policies.
"""

import numpy as np
import json
import pickle
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HumanAction:
    """Represents a human action with context."""
    action: int
    timestamp: float
    state_hash: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class HumanGameplay:
    """Represents a complete human gameplay session."""
    session_id: str
    player_id: str
    actions: List[HumanAction]
    total_reward: float
    session_metadata: Dict[str, Any]
    recording_timestamp: float

class HumanReplayPolicy:
    """
    Policy that replays recorded human gameplay.
    """
    
    def __init__(self, replay_sources: Union[str, List[str]], 
                 replay_mode: str = "sequential", 
                 fallback_policy=None,
                 state_matching: bool = True,
                 seed: Optional[int] = None):
        """
        Initialize human replay policy.
        
        Args:
            replay_sources: Path(s) to human replay files
            replay_mode: How to select replays ('sequential', 'random', 'weighted')
            fallback_policy: Policy to use when replay data is exhausted
            state_matching: Whether to try matching game states
            seed: Random seed for replay selection
        """
        self.rng = np.random.RandomState(seed)
        self.replay_mode = replay_mode
        self.fallback_policy = fallback_policy
        self.state_matching = state_matching
        
        # Load replay data
        self.replays = self._load_replays(replay_sources)
        
        # Replay state
        self.current_replay_idx = 0
        self.current_action_idx = 0
        self.replay_weights = self._compute_replay_weights()
        
        # Statistics
        self.actions_from_replay = 0
        self.actions_from_fallback = 0
        self.state_matches = 0
        self.state_mismatches = 0
        
        logger.info(f"Loaded {len(self.replays)} human replays")
        logger.info(f"Replay mode: {replay_mode}, State matching: {state_matching}")
    
    def _load_replays(self, replay_sources: Union[str, List[str]]) -> List[HumanGameplay]:
        """Load human replay data from files."""
        if isinstance(replay_sources, str):
            replay_sources = [replay_sources]
        
        replays = []
        
        for source in replay_sources:
            source_path = Path(source)
            
            if source_path.is_file():
                replays.extend(self._load_single_file(source_path))
            elif source_path.is_dir():
                # Load all replay files from directory
                for file_path in source_path.glob("*.json"):
                    replays.extend(self._load_single_file(file_path))
                for file_path in source_path.glob("*.pkl"):
                    replays.extend(self._load_single_file(file_path))
            else:
                logger.warning(f"Replay source not found: {source}")
        
        return replays
    
    def _load_single_file(self, file_path: Path) -> List[HumanGameplay]:
        """Load replays from a single file."""
        try:
            if file_path.suffix == '.json':
                return self._load_json_replays(file_path)
            elif file_path.suffix == '.pkl':
                return self._load_pickle_replays(file_path)
            else:
                logger.warning(f"Unknown file format: {file_path}")
                return []
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []
    
    def _load_json_replays(self, file_path: Path) -> List[HumanGameplay]:
        """Load replays from JSON format."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        replays = []
        
        # Handle different JSON formats
        if isinstance(data, list):
            # Array of replay sessions
            for session_data in data:
                replay = self._parse_session_data(session_data)
                if replay:
                    replays.append(replay)
        elif isinstance(data, dict):
            # Single replay session or structured format
            if 'sessions' in data:
                for session_data in data['sessions']:
                    replay = self._parse_session_data(session_data)
                    if replay:
                        replays.append(replay)
            else:
                replay = self._parse_session_data(data)
                if replay:
                    replays.append(replay)
        
        return replays
    
    def _load_pickle_replays(self, file_path: Path) -> List[HumanGameplay]:
        """Load replays from pickle format."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            return [replay for replay in data if isinstance(replay, HumanGameplay)]
        elif isinstance(data, HumanGameplay):
            return [data]
        else:
            logger.warning(f"Unexpected pickle format in {file_path}")
            return []
    
    def _parse_session_data(self, session_data: Dict) -> Optional[HumanGameplay]:
        """Parse session data into HumanGameplay object."""
        try:
            actions = []
            for action_data in session_data.get('actions', []):
                action = HumanAction(
                    action=action_data['action'],
                    timestamp=action_data.get('timestamp', 0.0),
                    state_hash=action_data.get('state_hash'),
                    confidence=action_data.get('confidence', 1.0),
                    metadata=action_data.get('metadata', {})
                )
                actions.append(action)
            
            return HumanGameplay(
                session_id=session_data.get('session_id', 'unknown'),
                player_id=session_data.get('player_id', 'unknown'),
                actions=actions,
                total_reward=session_data.get('total_reward', 0.0),
                session_metadata=session_data.get('metadata', {}),
                recording_timestamp=session_data.get('recording_timestamp', 0.0)
            )
        except Exception as e:
            logger.error(f"Failed to parse session data: {e}")
            return None
    
    def _compute_replay_weights(self) -> np.ndarray:
        """Compute weights for replay selection based on quality metrics."""
        if not self.replays:
            return np.array([])
        
        weights = []
        for replay in self.replays:
            # Weight based on total reward and action count
            reward_weight = max(0.1, replay.total_reward / 1000.0)  # Normalize rewards
            length_weight = min(2.0, len(replay.actions) / 100.0)  # Prefer longer sessions
            quality_weight = 1.0  # Could be based on player rating, etc.
            
            total_weight = reward_weight * length_weight * quality_weight
            weights.append(total_weight)
        
        return np.array(weights) / np.sum(weights) if weights else np.array([])
    
    def select_action(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        """
        Select action by replaying human gameplay.
        
        Args:
            state: Current game state (used for state matching if enabled)
            info: Additional game information
            
        Returns:
            Action from human replay or fallback policy
        """
        if not self.replays:
            return self._fallback_action(state, info)
        
        # Get current replay
        current_replay = self._get_current_replay()
        if not current_replay:
            return self._fallback_action(state, info)
        
        # Get current action from replay
        if self.current_action_idx >= len(current_replay.actions):
            # Replay exhausted, move to next or fallback
            self._advance_replay()
            return self._fallback_action(state, info)
        
        human_action = current_replay.actions[self.current_action_idx]
        
        # State matching if enabled
        if self.state_matching:
            if self._states_match(state, human_action.state_hash):
                self.state_matches += 1
                action = human_action.action
            else:
                self.state_mismatches += 1
                action = self._fallback_action(state, info)
        else:
            action = human_action.action
        
        # Advance replay position
        self.current_action_idx += 1
        self.actions_from_replay += 1
        
        return action
    
    def _get_current_replay(self) -> Optional[HumanGameplay]:
        """Get the current replay based on replay mode."""
        if not self.replays:
            return None
        
        if self.replay_mode == "sequential":
            if self.current_replay_idx < len(self.replays):
                return self.replays[self.current_replay_idx]
        
        elif self.replay_mode == "random":
            return self.rng.choice(self.replays)
        
        elif self.replay_mode == "weighted":
            if len(self.replay_weights) > 0:
                replay_idx = self.rng.choice(len(self.replays), p=self.replay_weights)
                return self.replays[replay_idx]
        
        return None
    
    def _advance_replay(self):
        """Advance to the next replay."""
        if self.replay_mode == "sequential":
            self.current_replay_idx += 1
            self.current_action_idx = 0
    
    def _states_match(self, current_state: Dict[str, Any], target_state_hash: Optional[str]) -> bool:
        """
        Check if current state matches the expected state from human replay.
        
        This is a simplified implementation - in practice you'd want more
        sophisticated state matching based on game-relevant features.
        """
        if not target_state_hash:
            return False
        
        # Simple state hash based on key features
        current_hash = self._compute_state_hash(current_state)
        return current_hash == target_state_hash
    
    def _compute_state_hash(self, state: Dict[str, Any]) -> str:
        """Compute a hash of the game state for matching."""
        # Simplified state hashing - focus on most important features
        key_features = {
            'ante': state.get('ante', 0),
            'money': state.get('money', 0),
            'hands_left': state.get('hands_left', 0),
            'discards': state.get('discards', 0),
            'phase': state.get('phase', 'unknown')
        }
        
        # Create hash from key features
        feature_str = json.dumps(key_features, sort_keys=True)
        return str(hash(feature_str))
    
    def _fallback_action(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        """Get action from fallback policy when replay is unavailable."""
        if self.fallback_policy:
            self.actions_from_fallback += 1
            return self.fallback_policy.select_action(state, info)
        else:
            # Default random action
            self.actions_from_fallback += 1
            return self.rng.randint(0, 10)  # Assuming 10 possible actions
    
    def get_policy_info(self) -> Dict[str, Any]:
        """Get information about this policy."""
        total_actions = self.actions_from_replay + self.actions_from_fallback
        
        return {
            'policy_type': 'human_replay',
            'replay_mode': self.replay_mode,
            'state_matching': self.state_matching,
            'total_replays': len(self.replays),
            'current_replay_idx': self.current_replay_idx,
            'current_action_idx': self.current_action_idx,
            'actions_from_replay': self.actions_from_replay,
            'actions_from_fallback': self.actions_from_fallback,
            'replay_usage_rate': self.actions_from_replay / max(total_actions, 1),
            'state_matches': self.state_matches,
            'state_mismatches': self.state_mismatches,
            'state_match_rate': self.state_matches / max(self.state_matches + self.state_mismatches, 1)
        }
    
    def reset_replay_position(self):
        """Reset replay position to beginning."""
        self.current_replay_idx = 0
        self.current_action_idx = 0
    
    def add_replay(self, replay: HumanGameplay):
        """Add a new human replay to the collection."""
        self.replays.append(replay)
        self.replay_weights = self._compute_replay_weights()


class HumanReplayRecorder:
    """
    Utility class for recording human gameplay for later replay.
    """
    
    def __init__(self, player_id: str = "unknown", session_id: Optional[str] = None):
        self.player_id = player_id
        self.session_id = session_id or f"session_{int(time.time())}"
        self.actions = []
        self.start_time = time.time()
        self.total_reward = 0.0
        self.session_metadata = {}
    
    def record_action(self, action: int, state: Dict[str, Any], 
                     confidence: float = 1.0, metadata: Dict[str, Any] = None):
        """Record a human action with context."""
        human_action = HumanAction(
            action=action,
            timestamp=time.time(),
            state_hash=self._compute_state_hash(state),
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.actions.append(human_action)
    
    def update_reward(self, reward: float):
        """Update the total reward for this session."""
        self.total_reward += reward
    
    def finalize_session(self, metadata: Dict[str, Any] = None) -> HumanGameplay:
        """Finalize and return the recorded gameplay session."""
        self.session_metadata.update(metadata or {})
        self.session_metadata['duration'] = time.time() - self.start_time
        
        return HumanGameplay(
            session_id=self.session_id,
            player_id=self.player_id,
            actions=self.actions.copy(),
            total_reward=self.total_reward,
            session_metadata=self.session_metadata.copy(),
            recording_timestamp=self.start_time
        )
    
    def _compute_state_hash(self, state: Dict[str, Any]) -> str:
        """Compute state hash for matching."""
        key_features = {
            'ante': state.get('ante', 0),
            'money': state.get('money', 0),
            'hands_left': state.get('hands_left', 0),
            'discards': state.get('discards', 0),
            'phase': state.get('phase', 'unknown')
        }
        
        feature_str = json.dumps(key_features, sort_keys=True)
        return str(hash(feature_str))
    
    def save_to_file(self, filename: str):
        """Save the recorded session to a file."""
        gameplay = self.finalize_session()
        
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump({
                    'session_id': gameplay.session_id,
                    'player_id': gameplay.player_id,
                    'actions': [
                        {
                            'action': action.action,
                            'timestamp': action.timestamp,
                            'state_hash': action.state_hash,
                            'confidence': action.confidence,
                            'metadata': action.metadata
                        }
                        for action in gameplay.actions
                    ],
                    'total_reward': gameplay.total_reward,
                    'metadata': gameplay.session_metadata,
                    'recording_timestamp': gameplay.recording_timestamp
                }, f, indent=2)
        
        elif filename.endswith('.pkl'):
            with open(filename, 'wb') as f:
                pickle.dump(gameplay, f)
        
        else:
            raise ValueError(f"Unsupported file format: {filename}")


# Factory function for creating human replay policies
def create_human_replay_policy(replay_sources: Union[str, List[str]], 
                              policy_type: str = "basic", 
                              **kwargs) -> HumanReplayPolicy:
    """
    Factory function to create human replay policies.
    
    Args:
        replay_sources: Path(s) to human replay files
        policy_type: Type of replay policy ('basic', 'state_matching', 'weighted')
        **kwargs: Additional arguments passed to policy constructor
        
    Returns:
        Human replay policy instance
    """
    if policy_type == "basic":
        return HumanReplayPolicy(
            replay_sources=replay_sources,
            replay_mode="sequential",
            state_matching=False,
            **kwargs
        )
    elif policy_type == "state_matching":
        return HumanReplayPolicy(
            replay_sources=replay_sources,
            replay_mode="sequential", 
            state_matching=True,
            **kwargs
        )
    elif policy_type == "weighted":
        return HumanReplayPolicy(
            replay_sources=replay_sources,
            replay_mode="weighted",
            state_matching=True,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown human replay policy type: {policy_type}")


if __name__ == "__main__":
    # Example usage and testing
    import tempfile
    import os
    
    print("Testing Human Replay Policy System:")
    
    # Create sample replay data
    print("1. Creating sample replay data...")
    
    recorder = HumanReplayRecorder(player_id="test_player", session_id="test_session_1")
    
    # Simulate recording some actions
    test_states = [
        {'ante': 1, 'money': 10, 'hands_left': 4, 'discards': 3, 'phase': 'play'},
        {'ante': 1, 'money': 12, 'hands_left': 3, 'discards': 3, 'phase': 'play'},
        {'ante': 1, 'money': 15, 'hands_left': 2, 'discards': 2, 'phase': 'play'},
    ]
    
    test_actions = [0, 2, 1]  # Play hand, select card, discard
    test_rewards = [5, 0, -1]
    
    for i, (state, action, reward) in enumerate(zip(test_states, test_actions, test_rewards)):
        recorder.record_action(action, state, confidence=0.9, metadata={'step': i})
        recorder.update_reward(reward)
    
    # Save replay to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_filename = f.name
    
    recorder.save_to_file(temp_filename)
    print(f"   Saved sample replay to: {temp_filename}")
    
    # Test loading and using the replay
    print("2. Testing replay policy...")
    
    try:
        # Create fallback policy for when replay is exhausted
        class SimpleFallback:
            def select_action(self, state, info):
                return 0  # Always play hand
        
        fallback = SimpleFallback()
        
        # Create replay policy
        replay_policy = HumanReplayPolicy(
            replay_sources=[temp_filename],
            replay_mode="sequential",
            fallback_policy=fallback,
            state_matching=True
        )
        
        print(f"   Loaded {len(replay_policy.replays)} replay(s)")
        
        # Test action selection
        for i, state in enumerate(test_states):
            action = replay_policy.select_action(state, {'phase': 'play'})
            print(f"   State {i+1}: Action = {action}")
        
        # Test exhausted replay (should use fallback)
        action = replay_policy.select_action(test_states[0], {'phase': 'play'})
        print(f"   Exhausted replay: Action = {action} (from fallback)")
        
        # Show policy statistics
        print("3. Policy statistics:")
        policy_info = replay_policy.get_policy_info()
        for key, value in policy_info.items():
            print(f"   {key}: {value}")
        
        print("✅ Human replay policy test completed successfully!")
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_filename)
        except:
            pass
    
    # Test different policy types
    print("\n4. Testing policy factory:")
    
    # Create another temporary replay file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_filename2 = f.name
    
    # Create a second replay session
    recorder2 = HumanReplayRecorder(player_id="test_player_2")
    for i, (state, action, reward) in enumerate(zip(test_states, [1, 0, 2], [2, 8, 1])):
        recorder2.record_action(action, state)
        recorder2.update_reward(reward)
    recorder2.save_to_file(temp_filename2)
    
    try:
        for policy_name in ['basic', 'state_matching', 'weighted']:
            test_policy = create_human_replay_policy(
                [temp_filename2], 
                policy_type=policy_name
            )
            action = test_policy.select_action(test_states[0], {'phase': 'play'})
            print(f"   {policy_name.title()} Policy: Action = {action}")
        
        print("✅ All human replay policy tests completed!")
        
    finally:
        try:
            os.unlink(temp_filename2)
        except:
            pass
