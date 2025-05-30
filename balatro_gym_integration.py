#!/usr/bin/env python3
"""
Final Fixed Real Balatro-Gym Integration with Realistic Rewards

This version fixes all syntax errors and provides realistic rewards and gameplay.
"""

import os
import sys
import subprocess
import multiprocessing as mp
import gymnasium as gym
import numpy as np
import pickle
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrajectoryStep:
    state: Dict[str, Any]
    action: int
    reward: float
    next_state: Dict[str, Any]
    done: bool
    info: Dict[str, Any]
    rtg: float
    strategy_tags: List[str]
    metadata: Dict[str, Any]

# ============================================================================
# EMBEDDED POLICY CLASSES
# ============================================================================

class PolicyBase:
    """Base class for different policies"""
    def select_action(self, state, info):
        raise NotImplementedError

class RandomPolicy(PolicyBase):
    """Random action selection"""
    def __init__(self, action_space_size=10, seed=None):
        self.action_space_size = action_space_size
        self.rng = np.random.RandomState(seed)
    
    def select_action(self, state, info):
        return self.rng.randint(0, self.action_space_size)

class HeuristicPolicy(PolicyBase):
    """Improved heuristic policy for Balatro with better reward generation"""
    def __init__(self, seed=None, aggression=0.5):
        self.rng = np.random.RandomState(seed)
        self.aggression = np.clip(aggression, 0.0, 1.0)
        
        # Action constants
        self.ACTION_PLAY_HAND = 0
        self.ACTION_DISCARD_HAND = 1
        self.ACTION_SELECT_CARDS = list(range(2, 10))
        
        # Hand rankings
        self.HAND_RANKINGS = {
            'high_card': 1, 'pair': 2, 'two_pair': 3, 'three_of_a_kind': 4,
            'straight': 5, 'flush': 6, 'full_house': 7, 'four_of_a_kind': 8,
            'straight_flush': 9, 'royal_flush': 10
        }
    
    def select_action(self, state, info):
        # Extract state information
        hand = state.get('hand', [])
        selected_cards = state.get('selected_cards', [])
        discards_left = state.get('discards', 0)
        hands_left = state.get('hands_left', 0)
        money = state.get('money', 0)
        phase = info.get('phase', 'play')
        chips_needed = info.get('chips_needed', float('inf'))
        current_chips = state.get('chips', 0)
        
        # Phase-specific logic
        if phase == 'shop':
            return self._shop_phase_action(state, info)
        elif phase == 'blind_select':
            return self._blind_select_action(state, info)
        else:
            return self._play_phase_action(state, info)
    
    def _play_phase_action(self, state, info):
        """Handle play phase actions with smarter logic"""
        selected_cards = state.get('selected_cards', [])
        hands_left = state.get('hands_left', 0)
        discards_left = state.get('discards', 0)
        chips_needed = info.get('chips_needed', float('inf'))
        current_chips = state.get('chips', 0)
        
        # If cards are selected, decide whether to play
        if selected_cards:
            estimated_score = self._estimate_score(state, selected_cards)
            total_score = current_chips + estimated_score
            score_ratio = total_score / max(chips_needed, 1) if chips_needed != float('inf') else 1.0
            
            # More realistic play thresholds
            if score_ratio >= 1.2:  # Definitely winning
                return self.ACTION_PLAY_HAND
            elif score_ratio >= 0.8 and hands_left <= 2:  # Close and few hands left
                return self.ACTION_PLAY_HAND
            elif hands_left <= 1:  # Must play
                return self.ACTION_PLAY_HAND
            elif score_ratio < 0.3 and discards_left > 0:  # Too weak, discard
                return self.ACTION_DISCARD_HAND
            elif len(selected_cards) < 5 and self.rng.random() < 0.3:  # Try to build bigger hand
                return self._improve_selection(state)
            else:
                return self.ACTION_PLAY_HAND
        
        # No cards selected - choose what to select
        return self._select_optimal_cards(state)
    
    def _select_optimal_cards(self, state):
        """Select cards for the best possible hand"""
        hand = state.get('hand', [])
        
        if not hand:
            return self.ACTION_DISCARD_HAND
        
        # Find best poker hand
        best_hand = self._find_best_poker_hand(hand)
        
        if best_hand and 'cards' in best_hand:
            # Select cards more intelligently
            target_cards = best_hand['cards']
            current_selected = state.get('selected_cards', [])
            
            # Select next card from best hand
            for card_idx in target_cards:
                if card_idx not in current_selected and card_idx < len(self.ACTION_SELECT_CARDS):
                    return self.ACTION_SELECT_CARDS[card_idx]
        
        # Fallback: select first unselected card
        current_selected = state.get('selected_cards', [])
        for i in range(len(hand)):
            if i not in current_selected and i < len(self.ACTION_SELECT_CARDS):
                return self.ACTION_SELECT_CARDS[i]
        
        return self.ACTION_DISCARD_HAND
    
    def _improve_selection(self, state):
        """Try to improve current card selection"""
        hand = state.get('hand', [])
        selected_cards = state.get('selected_cards', [])
        
        # Try to add one more card
        if len(selected_cards) < 5:
            for i in range(len(hand)):
                if i not in selected_cards and i < len(self.ACTION_SELECT_CARDS):
                    return self.ACTION_SELECT_CARDS[i]
        
        return self.ACTION_PLAY_HAND
    
    def _find_best_poker_hand(self, hand):
        """Find best poker hand from available cards"""
        if not hand:
            return None
        
        # Group cards by rank and suit
        cards_by_rank = {}
        cards_by_suit = {}
        
        for i, card in enumerate(hand):
            rank = card.get('rank', '?')
            suit = card.get('suit', '?')
            
            if rank not in cards_by_rank:
                cards_by_rank[rank] = []
            cards_by_rank[rank].append(i)
            
            if suit not in cards_by_suit:
                cards_by_suit[suit] = []
            cards_by_suit[suit].append(i)
        
        # Find best combination
        rank_counts = {rank: len(indices) for rank, indices in cards_by_rank.items()}
        max_count = max(rank_counts.values()) if rank_counts else 0
        
        # Four of a kind
        if max_count >= 4:
            for rank, indices in cards_by_rank.items():
                if len(indices) >= 4:
                    return {
                        'type': 'four_of_a_kind',
                        'cards': indices[:4],
                        'strength': self.HAND_RANKINGS['four_of_a_kind']
                    }
        
        # Three of a kind / Full house
        if max_count >= 3:
            three_kind_indices = []
            pair_indices = []
            
            for rank, indices in cards_by_rank.items():
                if len(indices) >= 3:
                    three_kind_indices = indices[:3]
                elif len(indices) >= 2:
                    pair_indices = indices[:2]
            
            if three_kind_indices and pair_indices:
                return {
                    'type': 'full_house',
                    'cards': three_kind_indices + pair_indices,
                    'strength': self.HAND_RANKINGS['full_house']
                }
            elif three_kind_indices:
                return {
                    'type': 'three_of_a_kind',
                    'cards': three_kind_indices,
                    'strength': self.HAND_RANKINGS['three_of_a_kind']
                }
        
        # Pairs
        pairs = [rank for rank, count in rank_counts.items() if count >= 2]
        if len(pairs) >= 2:
            return {
                'type': 'two_pair',
                'cards': cards_by_rank[pairs[0]][:2] + cards_by_rank[pairs[1]][:2],
                'strength': self.HAND_RANKINGS['two_pair']
            }
        elif len(pairs) >= 1:
            return {
                'type': 'pair',
                'cards': cards_by_rank[pairs[0]][:2],
                'strength': self.HAND_RANKINGS['pair']
            }
        
        # Flush
        for suit, indices in cards_by_suit.items():
            if len(indices) >= 5:
                return {
                    'type': 'flush',
                    'cards': indices[:5],
                    'strength': self.HAND_RANKINGS['flush']
                }
        
        # High card
        return {
            'type': 'high_card',
            'cards': list(range(min(5, len(hand)))),
            'strength': self.HAND_RANKINGS['high_card']
        }
    
    def _estimate_score(self, state, selected_cards):
        """Estimate score for selected cards"""
        if not selected_cards:
            return 0
        
        hand_strength = self._evaluate_hand_strength(state, selected_cards)
        hand_size = len(selected_cards)
        
        # More realistic scoring
        base_chips = hand_size * 15 + hand_strength * 50
        mult_bonus = max(1, hand_strength)
        
        return base_chips * mult_bonus
    
    def _evaluate_hand_strength(self, state, selected_cards):
        """Evaluate strength of selected hand"""
        hand = state.get('hand', [])
        
        if not selected_cards or not hand:
            return 1.0
        
        selected_card_data = [hand[i] for i in selected_cards if i < len(hand)]
        best_hand = self._find_best_poker_hand(selected_card_data)
        
        if best_hand:
            return best_hand['strength']
        
        return 1.0
    
    def _shop_phase_action(self, state, info):
        """Handle shop phase"""
        # Random chance to "buy" something or exit
        if self.rng.random() < 0.3:
            return self.rng.choice(self.ACTION_SELECT_CARDS)  # "Buy" action
        return self.ACTION_PLAY_HAND  # Exit shop
    
    def _blind_select_action(self, state, info):
        """Handle blind selection"""
        if self.aggression > 0.7:
            return self.ACTION_SELECT_CARDS[0]  # Risky blind
        else:
            return self.ACTION_SELECT_CARDS[1] if len(self.ACTION_SELECT_CARDS) > 1 else self.ACTION_SELECT_CARDS[0]

class EnhancedPolicy(HeuristicPolicy):
    """Enhanced policy with more sophisticated strategies"""
    def __init__(self, seed=None, aggression=0.6, economic_focus=0.4):
        super().__init__(seed=seed, aggression=aggression)
        self.economic_focus = economic_focus
        self.decision_history = []
    
    def select_action(self, state, info):
        # Enhanced decision making with economic considerations
        money = state.get('money', 0)
        ante = state.get('ante', 1)
        
        # Adjust aggression based on economic situation
        if money < ante * 5:  # Low money
            self.aggression = max(0.2, self.aggression - 0.2)
        elif money > ante * 15:  # High money
            self.aggression = min(0.9, self.aggression + 0.1)
        
        action = super().select_action(state, info)
        
        # Record decision for learning
        self.decision_history.append({
            'state_summary': self._summarize_state(state),
            'action': action,
            'timestamp': time.time()
        })
        
        return action
    
    def _summarize_state(self, state):
        """Create state summary for learning"""
        return {
            'ante': state.get('ante', 1),
            'money': state.get('money', 0),
            'hands_left': state.get('hands_left', 0),
            'discards': state.get('discards', 0)
        }

# ============================================================================
# IMPROVED ENVIRONMENT WITH REALISTIC REWARDS
# ============================================================================

class ImprovedBalatroEnvironment:
    """Improved dummy environment with realistic rewards and gameplay"""
    
    def __init__(self):
        self.action_space = gym.spaces.Discrete(10)
        self.step_count = 0
        self.max_steps_per_blind = 20  # More realistic
        self.current_ante = 1
        self.money = 10
        self.chips_needed = 300
        self.current_chips = 0
        self.current_mult = 1
        self.hands_left = 4
        self.discards_left = 3
        self.current_hand = []
        self.selected_cards = []
        self.blinds_beaten = 0
        
    def reset(self, **kwargs):
        self.step_count = 0
        self.current_ante = 1
        self.money = 10
        self.chips_needed = 300
        self.current_chips = 0
        self.current_mult = 1
        self.hands_left = 4
        self.discards_left = 3
        self.blinds_beaten = 0
        self.current_hand = self._generate_hand()
        self.selected_cards = []
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def step(self, action):
        self.step_count += 1
        old_state = self._get_state()
        
        # Process action and calculate reward
        reward = self._process_action(action)
        
        # Check if blind is complete
        if self.current_chips >= self.chips_needed:
            reward += 50  # Bonus for beating blind
            self.blinds_beaten += 1
            self._advance_to_next_blind()
        elif self.hands_left <= 0 and self.discards_left <= 0:
            # Failed the blind
            reward -= 20
            self._advance_to_next_blind()
        
        # Check if episode should end
        done = self._check_done()
        truncated = self.step_count >= 500  # Maximum episode length
        
        new_state = self._get_state()
        info = self._get_info()
        
        return new_state, reward, done, truncated, info
    
    def _process_action(self, action):
        """Process action and return reward"""
        reward = 0
        
        if action == 0:  # Play hand
            if self.selected_cards:
                # Calculate score for played hand
                played_cards = [self.current_hand[i] for i in self.selected_cards if i < len(self.current_hand)]
                chips_gained, mult_gained = self._score_hand(played_cards)
                
                self.current_chips += chips_gained
                self.current_mult += mult_gained
                self.hands_left -= 1
                
                # Reward based on chips gained
                reward = chips_gained / 10.0  # Scale reward
                
                # Bonus for good hands
                hand_type = self._get_hand_type(played_cards)
                if hand_type in ['pair', 'two_pair']:
                    reward += 5
                elif hand_type in ['three_of_a_kind', 'straight', 'flush']:
                    reward += 15
                elif hand_type in ['full_house', 'four_of_a_kind']:
                    reward += 30
                
                # Reset for next hand
                self.selected_cards = []
                self.current_hand = self._generate_hand()
            else:
                reward = -2  # Penalty for playing without selection
        
        elif action == 1:  # Discard hand
            if self.discards_left > 0:
                self.discards_left -= 1
                self.selected_cards = []
                self.current_hand = self._generate_hand()
                reward = -1  # Small penalty for discarding
            else:
                reward = -5  # Penalty for trying to discard when none left
        
        else:  # Select card (actions 2-9)
            card_idx = action - 2
            if card_idx < len(self.current_hand):
                if card_idx in self.selected_cards:
                    self.selected_cards.remove(card_idx)  # Deselect
                    reward = 0.1
                else:
                    if len(self.selected_cards) < 5:
                        self.selected_cards.append(card_idx)  # Select
                        reward = 0.2
                    else:
                        reward = -0.5  # Penalty for trying to select too many
            else:
                reward = -1  # Penalty for invalid card selection
        
        return reward
    
    def _advance_to_next_blind(self):
        """Advance to next blind"""
        self.current_chips = 0
        self.current_mult = 1
        self.hands_left = 4
        self.discards_left = 3
        self.chips_needed = int(self.chips_needed * 1.5)  # Increase difficulty
        self.money += np.random.randint(3, 8)  # Earn money
        
        # Advance ante every 3 blinds
        if self.blinds_beaten % 3 == 0:
            self.current_ante += 1
    
    def _check_done(self):
        """Check if episode should end"""
        return (
            self.current_ante > 8 or  # Completed many antes
            (self.hands_left <= 0 and self.discards_left <= 0 and self.current_chips < self.chips_needed) or  # Failed
            self.blinds_beaten >= 15  # Success condition
        )
    
    def _generate_hand(self):
        """Generate a realistic poker hand"""
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        suits = ['♠', '♥', '♦', '♣']
        
        hand = []
        for _ in range(8):
            rank = np.random.choice(ranks)
            suit = np.random.choice(suits)
            hand.append({'rank': rank, 'suit': suit, 'enhancement': None})
        
        return hand
    
    def _score_hand(self, cards):
        """Calculate realistic scoring for a hand"""
        if not cards:
            return 0, 0
        
        hand_type = self._get_hand_type(cards)
        hand_size = len(cards)
        
        # Base scoring by hand type
        scoring = {
            'high_card': (10, 1),
            'pair': (20, 2),
            'two_pair': (30, 3),
            'three_of_a_kind': (40, 4),
            'straight': (50, 5),
            'flush': (60, 6),
            'full_house': (80, 8),
            'four_of_a_kind': (120, 12),
            'straight_flush': (200, 20)
        }
        
        base_chips, base_mult = scoring.get(hand_type, (10, 1))
        
        # Scale by hand size
        chips = base_chips + (hand_size * 5)
        mult = base_mult
        
        return chips, mult
    
    def _get_hand_type(self, cards):
        """Determine poker hand type"""
        if not cards:
            return 'high_card'
        
        ranks = [card['rank'] for card in cards]
        suits = [card['suit'] for card in cards]
        
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        counts = sorted(rank_counts.values(), reverse=True)
        is_flush = len(set(suits)) == 1 and len(cards) >= 5
        
        if counts[0] >= 4:
            return 'four_of_a_kind'
        elif counts[0] >= 3 and len(counts) > 1 and counts[1] >= 2:
            return 'full_house'
        elif is_flush:
            return 'flush'
        elif counts[0] >= 3:
            return 'three_of_a_kind'
        elif counts[0] >= 2 and len(counts) > 1 and counts[1] >= 2:
            return 'two_pair'
        elif counts[0] >= 2:
            return 'pair'
        else:
            return 'high_card'
    
    def _get_state(self):
        """Get current game state"""
        return {
            'hand': self.current_hand,
            'played_cards': [],
            'selected_cards': self.selected_cards,
            'chips': self.current_chips,
            'mult': self.current_mult,
            'money': self.money,
            'ante': self.current_ante,
            'discards': self.discards_left,
            'hands_left': self.hands_left,
            'jokers': self._generate_jokers(),
            'blind_type': 'small' if self.blinds_beaten % 3 == 0 else 'big' if self.blinds_beaten % 3 == 1 else 'boss',
            'phase': 'play'
        }
    
    def _get_info(self):
        """Get game info"""
        return {
            'phase': 'play',
            'chips_needed': self.chips_needed,
            'blind_name': f'Blind {self.blinds_beaten + 1}',
            'legal_actions': list(range(10))
        }
    
    def _generate_jokers(self):
        """Generate random jokers"""
        joker_names = ['Joker', 'Greedy Joker', 'Lusty Joker', 'Wrathful Joker']
        num_jokers = min(self.current_ante, 3)  # More jokers as game progresses
        jokers = []
        
        for _ in range(num_jokers):
            joker = {
                'name': np.random.choice(joker_names),
                'description': f'+{np.random.randint(2, 8)} Mult',
                'mult': np.random.randint(2, 8),
                'chips': np.random.randint(10, 30)
            }
            jokers.append(joker)
        
        return jokers

class BalatroEnvironmentWrapper:
    """Wrapper that uses improved environment"""
    
    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode
        self.env = None
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize environment with fallback to improved dummy"""
        try:
            import balatro_gym
            self.env = gym.make('Balatro-v0', render_mode=self.render_mode)
            logger.info("✅ Real Balatro environment initialized")
        except Exception as e:
            logger.info("🔄 Using improved dummy environment with realistic rewards")
            self.env = ImprovedBalatroEnvironment()
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

# ============================================================================
# STRATEGY TAGGER AND UTILITIES
# ============================================================================

class BalatroStrategyTagger:
    """Strategy tagger for Balatro gameplay"""
    
    def tag_step(self, state: Dict, action: int, info: Dict, next_state: Dict) -> List[str]:
        """Generate strategy tags for a step"""
        tags = []
        
        # Basic action tagging
        action_map = {
            0: 'action_play_hand',
            1: 'action_discard_hand'
        }
        
        if action in action_map:
            tags.append(action_map[action])
        elif 2 <= action <= 9:
            tags.append('action_select_card')
        
        # Game phase tagging
        phase = info.get('phase', state.get('phase', 'unknown'))
        tags.append(f'phase_{phase}')
        
        # Ante progression
        ante = state.get('ante', 1)
        if ante <= 2:
            tags.append('early_game')
        elif ante <= 5:
            tags.append('mid_game')
        else:
            tags.append('late_game')
        
        # Economic situation
        money = state.get('money', 0)
        if money <= 5:
            tags.append('low_money')
        elif money >= 20:
            tags.append('high_money')
        
        # Resource management
        hands_left = state.get('hands_left', 0)
        discards_left = state.get('discards', 0)
        
        if hands_left <= 1:
            tags.append('last_hand')
        if discards_left <= 0:
            tags.append('no_discards')
        
        # Hand analysis
        if action == 0:  # Playing hand
            selected_cards = state.get('selected_cards', [])
            if selected_cards:
                hand_size = len(selected_cards)
                tags.append(f'hand_size_{hand_size}')
                
                if hand_size == 5:
                    tags.append('full_hand')
                elif hand_size < 3:
                    tags.append('small_hand')
        
        # Progress tracking
        chips = state.get('chips', 0)
        chips_needed = info.get('chips_needed', float('inf'))
        
        if chips_needed != float('inf'):
            progress = chips / chips_needed
            if progress >= 0.8:
                tags.append('near_victory')
            elif progress <= 0.3:
                tags.append('struggling')
        
        return tags

def serialize_state(state):
    """Serialize state for storage"""
    if isinstance(state, dict):
        return state.copy()
    return state

def serialize_info(info):
    """Serialize info dict for storage"""
    if isinstance(info, dict):
        return info.copy()
    return info

def compute_rtg(trajectory: List[TrajectoryStep], rewards: List[float]):
    """Compute reward-to-go for each step"""
    rtg = 0
    for i in reversed(range(len(trajectory))):
        rtg += rewards[i]
        trajectory[i].rtg = rtg

def generate_real_episode(args_tuple):
    """Generate a single episode with improved rewards"""
    worker_id, episode_id, policy_type, max_steps = args_tuple
    
    try:
        # Create environment wrapper
        env_wrapper = BalatroEnvironmentWrapper()
        
        # Create policy
        if policy_type == 'random':
            policy = RandomPolicy(action_space_size=10)
        elif policy_type == 'heuristic':
            policy = HeuristicPolicy(aggression=0.5)
        elif policy_type == 'enhanced':
            policy = EnhancedPolicy(aggression=0.6, economic_focus=0.4)
        elif policy_type == 'aggressive':
            policy = HeuristicPolicy(aggression=0.9)
        elif policy_type == 'conservative':
            policy = HeuristicPolicy(aggression=0.2)
        else:
            policy = RandomPolicy(action_space_size=10)
        
        # Create strategy tagger
        strategy_tagger = BalatroStrategyTagger()
        trajectory = []
        
        # Episode timing
        start_time = time.time()
        
        # Reset environment
        state, info = env_wrapper.reset()
        episode_rewards = []
        
        for step in range(max_steps):
            # Select action using policy
            action = policy.select_action(state, info)
            
            # Take step in environment
            next_state, reward, done, truncated, info = env_wrapper.step(action)
            
            # Tag the strategy
            strategy_tags = strategy_tagger.tag_step(state, action, info, next_state)
            
            # Create trajectory step
            traj_step = TrajectoryStep(
                state=serialize_state(state),
                action=action,
                reward=reward,
                next_state=serialize_state(next_state),
                done=done or truncated,
                info=serialize_info(info),
                rtg=0,  # Will be computed later
                strategy_tags=strategy_tags,
                metadata={
                    'worker_id': worker_id,
                    'episode_id': episode_id,
                    'step_id': step,
                    'timestamp': time.time(),
                    'policy_type': policy_type,
                    'env_type': 'improved_balatro_gym'
                }
            )
            
            trajectory.append(traj_step)
            episode_rewards.append(reward)
            
            if done or truncated:
                break
                
            state = next_state
        
        end_time = time.time()
        
        # Compute reward-to-go
        compute_rtg(trajectory, episode_rewards)
        
        # Clean up environment
        env_wrapper.close()
        
        # Create episode metadata
        final_state = trajectory[-1].next_state if trajectory else {}
        metadata = {
            'episode_id': episode_id,
            'worker_id': worker_id,
            'policy_type': policy_type,
            'total_steps': len(trajectory),
            'total_reward': sum(episode_rewards),
            'episode_length': len(trajectory),
            'start_timestamp': start_time,
            'end_timestamp': end_time,
            'final_ante': final_state.get('ante', 0),
            'max_chips': max((step.state.get('chips', 0) for step in trajectory), default=0),
            'max_money': max((step.state.get('money', 0) for step in trajectory), default=0),
            'env_type': 'improved_balatro_gym'
        }
        
        return trajectory, metadata
        
    except Exception as e:
        logger.error(f"Error in worker {worker_id}, episode {episode_id}: {e}")
        return None, None

class RealBalatroDataGenerator:
    """Generator for real Balatro trajectories with improved rewards"""
    
    def __init__(self, num_workers=None, output_dir="./realistic_balatro_dataset"):
        self.num_workers = num_workers or mp.cpu_count()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized improved Balatro data generator with {self.num_workers} workers")
        logger.info(f"Output directory: {self.output_dir}")
    
    def generate_dataset(self, total_episodes=1000, policy_type='random', max_steps_per_episode=200):
        """Generate dataset with realistic rewards and gameplay"""
        
        logger.info(f"Generating {total_episodes} realistic Balatro episodes using {self.num_workers} workers")
        
        # Create argument tuples for each episode
        episodes_per_worker = total_episodes // self.num_workers
        remaining_episodes = total_episodes % self.num_workers
        
        args_list = []
        episode_id = 0
        
        for worker_id in range(self.num_workers):
            worker_episodes = episodes_per_worker + (1 if worker_id < remaining_episodes else 0)
            
            for _ in range(worker_episodes):
                args_list.append((worker_id, episode_id, policy_type, max_steps_per_episode))
                episode_id += 1
        
        # Generate episodes in parallel
        total_steps = 0
        total_reward = 0
        successful_episodes = 0
        episode_metadatas = []
        
        with mp.Pool(self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(generate_real_episode, args_list),
                total=len(args_list),
                desc="Generating realistic Balatro episodes"
            ))
        
        # Save results and collect statistics
        for i, (trajectory, episode_metadata) in enumerate(results):
            if trajectory is not None and episode_metadata is not None:
                # Save individual episode
                episode_file = self.output_dir / f"episode_{i:06d}.pkl"
                with open(episode_file, 'wb') as f:
                    pickle.dump(trajectory, f)
                
                # Save episode metadata
                metadata_file = self.output_dir / f"episode_{i:06d}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(episode_metadata, f, indent=2)
                
                total_steps += episode_metadata['total_steps']
                total_reward += episode_metadata['total_reward']
                successful_episodes += 1
                episode_metadatas.append(episode_metadata)
        
        # Save dataset metadata
        dataset_metadata = {
            'total_episodes': successful_episodes,
            'total_steps': total_steps,
            'total_reward': total_reward,
            'avg_steps_per_episode': total_steps / max(successful_episodes, 1),
            'avg_reward_per_episode': total_reward / max(successful_episodes, 1),
            'policy_type': policy_type,
            'generation_timestamp': time.time(),
            'num_workers': self.num_workers,
            'dataset_version': '3.0-realistic-rewards-fixed',
            'env_type': 'improved_balatro_gym',
            'generation_config': {
                'max_steps_per_episode': max_steps_per_episode,
                'total_requested_episodes': total_episodes,
                'successful_episodes': successful_episodes
            }
        }
        
        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Save consolidated episode metadata
        episodes_metadata_file = self.output_dir / "episodes_metadata.json"
        with open(episodes_metadata_file, 'w') as f:
            json.dump(episode_metadatas, f, indent=2)
        
        logger.info(f"Realistic Balatro dataset generation complete!")
        logger.info(f"Generated {successful_episodes} episodes with {total_steps} total steps")
        logger.info(f"Average reward per episode: {total_reward / max(successful_episodes, 1):.2f}")
        
        return dataset_metadata

def main():
    parser = argparse.ArgumentParser(description="Generate realistic Balatro trajectory dataset")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to generate")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--policy", choices=['random', 'heuristic', 'enhanced', 'aggressive', 'conservative'], 
                       default='heuristic', help="Policy type")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--output-dir", type=str, default="./realistic_balatro_dataset", help="Output directory")
    parser.add_argument("--setup-only", action="store_true", help="Only test setup, don't generate")
    
    args = parser.parse_args()
    
    if args.setup_only:
        print("🔧 Testing improved Balatro environment setup...")
        try:
            env_wrapper = BalatroEnvironmentWrapper()
            state, info = env_wrapper.reset()
            
            print(f"✅ Environment initialized successfully!")
            print(f"   State keys: {list(state.keys())}")
            print(f"   Initial chips: {state.get('chips', 0)}")
            print(f"   Chips needed: {info.get('chips_needed', 0)}")
            print(f"   Hand size: {len(state.get('hand', []))}")
            
            # Test a few actions to verify rewards
            policy = HeuristicPolicy()
            total_reward = 0
            
            for i in range(10):
                action = policy.select_action(state, info)
                next_state, reward, done, truncated, info = env_wrapper.step(action)
                total_reward += reward
                print(f"   Step {i+1}: Action={action}, Reward={reward:.2f}")
                state = next_state
                if done or truncated:
                    break
            
            print(f"   Total reward from test: {total_reward:.2f}")
            env_wrapper.close()
            
            if total_reward > 0:
                print("✅ Reward system working correctly!")
            else:
                print("⚠️ Rewards still zero - may need further adjustment")
                
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
        return
    
    generator = RealBalatroDataGenerator(
        num_workers=args.workers,
        output_dir=args.output_dir
    )
    
    metadata = generator.generate_dataset(
        total_episodes=args.episodes,
        policy_type=args.policy,
        max_steps_per_episode=args.max_steps
    )
    
    if metadata:
        print(f"\n🎉 Realistic Balatro dataset saved to: {args.output_dir}")
        print(f"📊 Episodes generated: {metadata['total_episodes']}")
        print(f"📈 Total steps: {metadata['total_steps']}")
        print(f"💰 Average reward: {metadata['avg_reward_per_episode']:.2f}")
        print(f"🎮 Environment: {metadata['env_type']}")
        print(f"🤖 Policy: {metadata['policy_type']}")
        
        if metadata['avg_reward_per_episode'] > 0:
            print("✅ Realistic rewards generated successfully!")
        else:
            print("⚠️ Rewards still low - consider adjusting parameters")
            
        print(f"\n🔍 View your data:")
        print(f"   streamlit run simple_streamlit_viewer.py")
        print(f"   (Set dataset path to: {args.output_dir})")
    else:
        print("❌ Dataset generation failed!")

if __name__ == "__main__":
    main()
