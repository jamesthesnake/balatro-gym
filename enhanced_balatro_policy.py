"""
Enhanced Balatro Policy for Real Gameplay

This policy is specifically designed to work with real balatro-gym environments
and includes sophisticated Balatro-specific strategies.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedBalatroPolicy:
    """
    Advanced policy for real Balatro gameplay that understands:
    - Poker hand rankings and scoring
    - Joker synergies and interactions  
    - Economic management and shop decisions
    - Blind-specific strategies
    - Risk assessment and optimal play timing
    """
    
    def __init__(self, 
                 aggression: float = 0.6, 
                 economic_focus: float = 0.4,
                 risk_tolerance: float = 0.5,
                 seed: Optional[int] = None):
        """
        Initialize enhanced Balatro policy.
        
        Args:
            aggression: How aggressively to play (0.0 = very conservative, 1.0 = very aggressive)
            economic_focus: How much to prioritize money management (0.0 = ignore, 1.0 = prioritize)
            risk_tolerance: Willingness to take risks (0.0 = never risk, 1.0 = always risk)
            seed: Random seed for reproducibility
        """
        self.aggression = np.clip(aggression, 0.0, 1.0)
        self.economic_focus = np.clip(economic_focus, 0.0, 1.0)
        self.risk_tolerance = np.clip(risk_tolerance, 0.0, 1.0)
        self.rng = np.random.RandomState(seed)
        
        # Poker hand rankings (Balatro scoring)
        self.hand_rankings = {
            'high_card': 1,
            'pair': 2,
            'two_pair': 3,
            'three_of_a_kind': 4,
            'straight': 5,
            'flush': 6,
            'full_house': 7,
            'four_of_a_kind': 8,
            'straight_flush': 9,
            'royal_flush': 10,
            'five_of_a_kind': 11,  # Balatro-specific
            'flush_house': 12,     # Balatro-specific
            'flush_five': 13       # Balatro-specific
        }
        
        # Decision history for learning
        self.decision_history = []
        self.performance_metrics = {
            'hands_played': 0,
            'hands_won': 0,
            'money_earned': 0,
            'blinds_beaten': 0
        }
        
        logger.info(f"Enhanced Balatro policy initialized:")
        logger.info(f"  Aggression: {self.aggression:.2f}")
        logger.info(f"  Economic Focus: {self.economic_focus:.2f}")
        logger.info(f"  Risk Tolerance: {self.risk_tolerance:.2f}")
    
    def select_action(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        """
        Select optimal action based on current game state and sophisticated Balatro strategies.
        """
        # Analyze current game situation
        game_analysis = self._analyze_game_state(state, info)
        
        # Determine current phase and act accordingly
        phase = info.get('phase', state.get('phase', 'play'))
        
        if phase == 'shop':
            return self._shop_phase_action(state, info, game_analysis)
        elif phase == 'blind_select':
            return self._blind_select_action(state, info, game_analysis)
        else:  # Play phase
            return self._play_phase_action(state, info, game_analysis)
    
    def _analyze_game_state(self, state: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive analysis of the current game state.
        """
        analysis = {}
        
        # Basic game state
        analysis['hand'] = state.get('hand', [])
        analysis['selected_cards'] = state.get('selected_cards', [])
        analysis['played_cards'] = state.get('played_cards', [])
        analysis['money'] = state.get('money', 0)
        analysis['chips'] = state.get('chips', 0)
        analysis['mult'] = state.get('mult', 1)
        analysis['ante'] = state.get('ante', 1)
        analysis['discards'] = state.get('discards', 0)
        analysis['hands_left'] = state.get('hands_left', 0)
        analysis['jokers'] = state.get('jokers', [])
        
        # Target and pressure analysis
        analysis['chips_needed'] = info.get('chips_needed', float('inf'))
        analysis['pressure_level'] = self._calculate_pressure_level(analysis)
        analysis['score_gap'] = max(0, analysis['chips_needed'] - analysis['chips'])
        
        # Hand analysis
        analysis['best_hand'] = self._find_best_possible_hand(analysis['hand'])
        analysis['selected_hand_value'] = self._evaluate_selected_hand(state)
        analysis['hand_potential'] = self._assess_hand_potential(analysis['hand'])
        
        # Economic analysis
        analysis['economic_situation'] = self._assess_economic_situation(analysis)
        
        # Joker synergy analysis
        analysis['joker_synergies'] = self._analyze_joker_synergies(analysis['jokers'], analysis['hand'])
        
        # Strategic recommendations
        analysis['recommended_strategy'] = self._determine_strategy(analysis)
        
        return analysis
    
    def _play_phase_action(self, state: Dict[str, Any], info: Dict[str, Any], analysis: Dict[str, Any]) -> int:
        """
        Determine best action during play phase.
        """
        selected_cards = analysis['selected_cards']
        hands_left = analysis['hands_left']
        discards = analysis['discards']
        pressure_level = analysis['pressure_level']
        
        # If we have cards selected, decide whether to play them
        if selected_cards:
            should_play = self._should_play_selected_hand(analysis)
            if should_play:
                self.performance_metrics['hands_played'] += 1
                return 0  # Play hand
            else:
                # Try to improve selection or discard
                if discards > 0 and pressure_level < 0.8:
                    return 1  # Discard and try again
                else:
                    # Try to optimize current selection
                    optimization_action = self._optimize_hand_selection(analysis)
                    if optimization_action is not None:
                        return optimization_action
                    else:
                        return 0  # Play what we have
        
        # No cards selected - choose what to select
        return self._select_optimal_cards(analysis)
    
    def _should_play_selected_hand(self, analysis: Dict[str, Any]) -> bool:
        """
        Decide whether to play the currently selected hand.
        """
        selected_value = analysis['selected_hand_value']
        chips_needed = analysis['chips_needed']
        hands_left = analysis['hands_left']
        pressure_level = analysis['pressure_level']
        
        if chips_needed == float('inf'):
            return True  # Play if no specific target
        
        # Calculate score ratio
        estimated_score = self._estimate_hand_score(analysis)
        score_ratio = estimated_score / chips_needed
        
        # Adjust thresholds based on personality and situation
        base_threshold = 1.0 - (self.aggression * 0.4)  # 0.6 to 1.0
        
        # Modify threshold based on pressure
        if pressure_level > 0.8:  # High pressure
            threshold = base_threshold * 0.7
        elif pressure_level < 0.3:  # Low pressure  
            threshold = base_threshold * 1.3
        else:
            threshold = base_threshold
        
        # Emergency situations
        if hands_left <= 1:
            return True  # Must play
        
        if hands_left == 2 and score_ratio >= 0.6:
            return True  # Good enough with backup
        
        return score_ratio >= threshold
    
    def _select_optimal_cards(self, analysis: Dict[str, Any]) -> int:
        """
        Select the optimal cards to build the best possible hand.
        """
        hand = analysis['hand']
        best_hand = analysis['best_hand']
        joker_synergies = analysis['joker_synergies']
        
        if not hand:
            return 1  # Discard if no hand
        
        # Find the best combination considering jokers
        target_cards = self._find_target_card_combination(hand, joker_synergies)
        
        if target_cards:
            # Select the first card of our target combination
            # In a real implementation, you'd want to select all at once
            for card_idx in target_cards:
                if card_idx < 8:  # Valid card selection action
                    return card_idx + 2  # Actions 2-9 for card selection
        
        # Fallback: select based on best hand analysis
        if best_hand and 'cards' in best_hand:
            for card_idx in best_hand['cards'][:1]:  # Select first card
                if card_idx < 8:
                    return card_idx + 2
        
        # Last resort: select first card
        return 2
    
    def _find_target_card_combination(self, hand: List[Dict], joker_synergies: Dict) -> List[int]:
        """
        Find the best card combination considering joker synergies.
        """
        if not hand:
            return []
        
        # For simplicity, return the best poker hand indices
        # In a full implementation, this would consider joker effects
        best_combo = self._find_best_poker_combination(hand)
        return best_combo.get('indices', [])
    
    def _find_best_poker_combination(self, hand: List[Dict]) -> Dict:
        """
        Find the best poker hand combination from available cards.
        """
        if not hand:
            return {}
        
        # Analyze all possible 5-card combinations
        best_combination = {'type': 'high_card', 'strength': 1, 'indices': [0]}
        
        # Group cards by rank and suit
        rank_groups = {}
        suit_groups = {}
        
        for i, card in enumerate(hand):
            rank = card.get('rank', '?')
            suit = card.get('suit', '?')
            
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(i)
            
            if suit not in suit_groups:
                suit_groups[suit] = []
            suit_groups[suit].append(i)
        
        # Check for pairs, three of a kind, etc.
        rank_counts = {rank: len(indices) for rank, indices in rank_groups.items()}
        
        # Find best combination
        max_count = max(rank_counts.values()) if rank_counts else 0
        
        if max_count >= 4:
            # Four of a kind
            for rank, indices in rank_groups.items():
                if len(indices) >= 4:
                    return {
                        'type': 'four_of_a_kind',
                        'strength': self.hand_rankings['four_of_a_kind'],
                        'indices': indices[:4]
                    }
        
        elif max_count >= 3:
            # Three of a kind or full house
            three_kind_indices = []
            pair_indices = []
            
            for rank, indices in rank_groups.items():
                if len(indices) >= 3:
                    three_kind_indices = indices[:3]
                elif len(indices) >= 2:
                    pair_indices = indices[:2]
            
            if three_kind_indices and pair_indices:
                return {
                    'type': 'full_house',
                    'strength': self.hand_rankings['full_house'],
                    'indices': three_kind_indices + pair_indices
                }
            elif three_kind_indices:
                return {
                    'type': 'three_of_a_kind',
                    'strength': self.hand_rankings['three_of_a_kind'],
                    'indices': three_kind_indices
                }
        
        elif max_count >= 2:
            # Pairs
            pair_indices = []
            for rank, indices in rank_groups.items():
                if len(indices) >= 2:
                    pair_indices.extend(indices[:2])
                    if len(pair_indices) >= 4:  # Two pair
                        return {
                            'type': 'two_pair',
                            'strength': self.hand_rankings['two_pair'],
                            'indices': pair_indices[:4]
                        }
            
            if pair_indices:
                return {
                    'type': 'pair',
                    'strength': self.hand_rankings['pair'],
                    'indices': pair_indices[:2]
                }
        
        # Check for flush
        for suit, indices in suit_groups.items():
            if len(indices) >= 5:
                return {
                    'type': 'flush',
                    'strength': self.hand_rankings['flush'],
                    'indices': indices[:5]
                }
        
        # High card fallback
        return {
            'type': 'high_card',
            'strength': self.hand_rankings['high_card'],
            'indices': [0]
        }
    
    def _calculate_pressure_level(self, analysis: Dict[str, Any]) -> float:
        """
        Calculate how much pressure the player is under (0.0 = no pressure, 1.0 = maximum pressure).
        """
        hands_left = analysis['hands_left']
        discards = analysis['discards']
        chips_needed = analysis['chips_needed']
        current_chips = analysis['chips']
        
        if chips_needed == float('inf'):
            return 0.0
        
        # Resource pressure
        resource_pressure = 1.0 - ((hands_left + discards) / 8.0)  # Assuming max 4 hands + 4 discards
        
        # Score pressure
        score_gap = max(0, chips_needed - current_chips)
        score_pressure = min(1.0, score_gap / chips_needed)
        
        return np.clip((resource_pressure + score_pressure) / 2.0, 0.0, 1.0)
    
    def _find_best_possible_hand(self, hand: List[Dict]) -> Dict:
        """
        Find the best possible poker hand from current cards.
        """
        return self._find_best_poker_combination(hand)
    
    def _evaluate_selected_hand(self, state: Dict[str, Any]) -> float:
        """
        Evaluate the value of currently selected cards.
        """
        selected_cards = state.get('selected_cards', [])
        hand = state.get('hand', [])
        
        if not selected_cards or not hand:
            return 0.0
        
        selected_card_data = [hand[i] for i in selected_cards if i < len(hand)]
        best_hand = self._find_best_poker_combination(selected_card_data)
        
        return best_hand.get('strength', 1)
    
    def _assess_hand_potential(self, hand: List[Dict]) -> Dict[str, Any]:
        """
        Assess the potential of the current hand for various poker combinations.
        """
        potential = {
            'pairs': 0,
            'three_of_a_kind': 0,
            'straight': 0,
            'flush': 0,
            'high_cards': 0
        }
        
        if not hand:
            return potential
        
        # Count ranks and suits
        ranks = [card.get('rank', '?') for card in hand]
        suits = [card.get('suit', '?') for card in hand]
        
        rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
        suit_counts = {suit: suits.count(suit) for suit in set(suits)}
        
        # Assess potential
        potential['pairs'] = sum(1 for count in rank_counts.values() if count >= 2)
        potential['three_of_a_kind'] = sum(1 for count in rank_counts.values() if count >= 3)
        potential['flush'] = max(suit_counts.values()) if suit_counts else 0
        potential['high_cards'] = len([rank for rank in ranks if rank in ['A', 'K', 'Q', 'J']])
        
        return potential
    
    def _assess_economic_situation(self, analysis: Dict[str, Any]) -> str:
        """
        Assess the current economic situation.
        """
        money = analysis['money']
        ante = analysis['ante']
        
        # Adjust expectations based on game progression
        expected_money = ante * 10  # Rough heuristic
        
        if money >= expected_money * 1.5:
            return 'wealthy'
        elif money >= expected_money:
            return 'comfortable'
        elif money >= expected_money * 0.5:
            return 'tight'
        else:
            return 'poor'
    
    def _analyze_joker_synergies(self, jokers: List[Dict], hand: List[Dict]) -> Dict[str, Any]:
        """
        Analyze synergies between jokers and current hand.
        """
        synergies = {
            'mult_bonus': 0,
            'chip_bonus': 0,
            'special_effects': [],
            'recommended_plays': []
        }
        
        for joker in jokers:
            if isinstance(joker, dict):
                # Add basic bonuses
                synergies['mult_bonus'] += joker.get('mult', 0)
                synergies['chip_bonus'] += joker.get('chips', 0)
                
                # Analyze special effects (simplified)
                joker_name = joker.get('name', '').lower()
                if 'pair' in joker_name and self._has_pair(hand):
                    synergies['special_effects'].append('pair_bonus')
                if 'flush' in joker_name and self._has_flush_potential(hand):
                    synergies['special_effects'].append('flush_bonus')
        
        return synergies
    
    def _has_pair(self, hand: List[Dict]) -> bool:
        """Check if hand has a pair."""
        if not hand:
            return False
        ranks = [card.get('rank', '?') for card in hand]
        rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
        return any(count >= 2 for count in rank_counts.values())
    
    def _has_flush_potential(self, hand: List[Dict]) -> bool:
        """Check if hand has flush potential."""
        if not hand:
            return False
        suits = [card.get('suit', '?') for card in hand]
        suit_counts = {suit: suits.count(suit) for suit in set(suits)}
        return any(count >= 3 for count in suit_counts.values())
    
    def _determine_strategy(self, analysis: Dict[str, Any]) -> str:
        """
        Determine the recommended strategy based on current situation.
        """
        pressure_level = analysis['pressure_level']
        economic_situation = analysis['economic_situation']
        hand_potential = analysis['hand_potential']
        ante = analysis['ante']
        
        # Early game strategy
        if ante <= 2:
            if economic_situation in ['poor', 'tight']:
                return 'economy_focus'
            else:
                return 'hand_building'
        
        # Mid game strategy
        elif ante <= 5:
            if pressure_level > 0.7:
                return 'aggressive_scoring'
            elif hand_potential['pairs'] >= 2:
                return 'pair_focus'
            else:
                return 'balanced'
        
        # Late game strategy
        else:
            if pressure_level > 0.8:
                return 'desperation'
            else:
                return 'high_value_hands'
    
    def _estimate_hand_score(self, analysis: Dict[str, Any]) -> float:
        """
        Estimate the score that would be achieved by playing selected cards.
        """
        selected_value = analysis['selected_hand_value']
        joker_synergies = analysis['joker_synergies']
        chips = analysis['chips']
        mult = analysis['mult']
        
        # Base scoring estimation
        base_chips = selected_value * 20  # Rough estimation
        base_mult = max(1, selected_value)
        
        # Add joker bonuses
        total_chips = chips + base_chips + joker_synergies['chip_bonus']
        total_mult = mult + base_mult + joker_synergies['mult_bonus']
        
        # Apply special effects multiplier
        special_multiplier = 1.0 + (len(joker_synergies['special_effects']) * 0.5)
        
        return total_chips * total_mult * special_multiplier
    
    def _optimize_hand_selection(self, analysis: Dict[str, Any]) -> Optional[int]:
        """
        Try to optimize the current hand selection by adding/removing cards.
        """
        hand = analysis['hand']
        selected_cards = analysis['selected_cards']
        
        if not hand:
            return None
        
        # Try to improve by selecting one more card
        for i in range(len(hand)):
            if i not in selected_cards and len(selected_cards) < 5:
                if i < 8:  # Valid selection action
                    return i + 2
        
        # Try to improve by deselecting a card
        if len(selected_cards) > 1:
            # Deselect the "worst" card (simplified heuristic)
            worst_card_idx = max(selected_cards)  # Simple heuristic
            if worst_card_idx < 8:
                return worst_card_idx + 2  # Toggle selection
        
        return None
    
    def _shop_phase_action(self, state: Dict[str, Any], info: Dict[str, Any], analysis: Dict[str, Any]) -> int:
        """
        Determine best action during shop phase.
        """
        money = analysis['money']
        economic_situation = analysis['economic_situation']
        ante = analysis['ante']
        
        # Shop items analysis (simplified - in real implementation would analyze actual items)
        if 'shop_items' in state:
            shop_items = state['shop_items']
            # Analyze each item and decide if it's worth buying
            for item in shop_items:
                if self._should_buy_item(item, analysis):
                    # Return action to buy this item (implementation depends on balatro-gym interface)
                    return 2  # Placeholder for buy action
        
        # Economic decision making
        if economic_situation == 'wealthy':
            # Can afford to buy useful items
            spending_threshold = money * 0.7
        elif economic_situation == 'comfortable':
            # Moderate spending
            spending_threshold = money * 0.4
        else:
            # Conservative spending
            spending_threshold = money * 0.2
        
        # Skip shop if nothing good or can't afford
        return 0  # Exit shop (play hand action)
    
    def _should_buy_item(self, item: Dict[str, Any], analysis: Dict[str, Any]) -> bool:
        """
        Decide whether to buy a specific shop item.
        """
        if not isinstance(item, dict):
            return False
        
        item_cost = item.get('cost', float('inf'))
        money = analysis['money']
        
        if item_cost > money:
            return False  # Can't afford
        
        # Simplified item evaluation
        item_type = item.get('type', 'unknown')
        item_value = self._evaluate_item_value(item, analysis)
        
        # Economic threshold based on money situation
        economic_situation = analysis['economic_situation']
        if economic_situation == 'wealthy':
            threshold = item_cost * 0.8
        elif economic_situation == 'comfortable':
            threshold = item_cost * 1.2
        else:
            threshold = item_cost * 2.0
        
        return item_value >= threshold
    
    def _evaluate_item_value(self, item: Dict[str, Any], analysis: Dict[str, Any]) -> float:
        """
        Evaluate the value of a shop item.
        """
        item_type = item.get('type', 'unknown')
        
        # Simplified evaluation - in real implementation would be much more sophisticated
        base_values = {
            'joker': 100,
            'tarot': 50,
            'planet': 75,
            'spectral': 60,
            'voucher': 80,
            'booster': 40
        }
        
        base_value = base_values.get(item_type, 30)
        
        # Adjust based on synergies with current build
        synergy_bonus = self._calculate_item_synergy(item, analysis)
        
        return base_value + synergy_bonus
    
    def _calculate_item_synergy(self, item: Dict[str, Any], analysis: Dict[str, Any]) -> float:
        """
        Calculate how well an item synergizes with current build.
        """
        # Simplified synergy calculation
        synergy = 0.0
        
        if item.get('type') == 'joker':
            # Joker synergies
            joker_effect = item.get('effect', '')
            current_jokers = analysis['jokers']
            
            # Prefer jokers that complement existing ones
            if len(current_jokers) < 3:  # Room for more jokers
                synergy += 20
            
            # Specific synergy checks (simplified)
            if 'mult' in joker_effect.lower():
                synergy += 15
            if 'chip' in joker_effect.lower():
                synergy += 10
        
        return synergy
    
    def _blind_select_action(self, state: Dict[str, Any], info: Dict[str, Any], analysis: Dict[str, Any]) -> int:
        """
        Determine best action during blind selection phase.
        """
        available_blinds = state.get('blind_options', [])
        risk_tolerance = self.risk_tolerance
        pressure_level = analysis['pressure_level']
        
        if not available_blinds:
            return 2  # Default selection
        
        # Analyze each blind option
        best_blind_idx = 0
        best_score = -float('inf')
        
        for i, blind in enumerate(available_blinds):
            score = self._evaluate_blind_choice(blind, analysis)
            if score > best_score:
                best_score = score
                best_blind_idx = i
        
        # Return action to select the best blind
        return 2 + best_blind_idx  # Assuming actions 2+ for blind selection
    
    def _evaluate_blind_choice(self, blind: Dict[str, Any], analysis: Dict[str, Any]) -> float:
        """
        Evaluate how good a blind choice is.
        """
        if not isinstance(blind, dict):
            return 0.0
        
        reward = blind.get('reward', 0)
        difficulty = blind.get('difficulty', 1.0)
        special_effects = blind.get('effects', [])
        
        # Base evaluation
        score = reward / max(difficulty, 0.1)
        
        # Adjust for risk tolerance
        risk_penalty = difficulty * (1.0 - self.risk_tolerance) * 10
        score -= risk_penalty
        
        # Penalty for harmful effects
        for effect in special_effects:
            if self._is_harmful_effect(effect, analysis):
                score -= 20
        
        return score
    
    def _is_harmful_effect(self, effect: str, analysis: Dict[str, Any]) -> bool:
        """
        Determine if a blind effect is harmful given current situation.
        """
        # Simplified harmful effect detection
        harmful_keywords = ['debuff', 'disable', 'destroy', 'reduce', 'prevent']
        return any(keyword in effect.lower() for keyword in harmful_keywords)
    
    def update_performance(self, reward: float, info: Dict[str, Any]):
        """
        Update performance metrics based on game outcomes.
        """
        if reward > 0:
            self.performance_metrics['money_earned'] += reward
        
        if info.get('blind_beaten', False):
            self.performance_metrics['blinds_beaten'] += 1
        
        if info.get('hand_won', False):
            self.performance_metrics['hands_won'] += 1
    
    def get_policy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about this policy.
        """
        win_rate = 0.0
        if self.performance_metrics['hands_played'] > 0:
            win_rate = self.performance_metrics['hands_won'] / self.performance_metrics['hands_played']
        
        return {
            'policy_type': 'enhanced_balatro',
            'aggression': self.aggression,
            'economic_focus': self.economic_focus,
            'risk_tolerance': self.risk_tolerance,
            'performance_metrics': self.performance_metrics.copy(),
            'win_rate': win_rate,
            'decisions_made': len(self.decision_history)
        }
    
    def reset_performance(self):
        """Reset performance metrics for a new session."""
        self.performance_metrics = {
            'hands_played': 0,
            'hands_won': 0,
            'money_earned': 0,
            'blinds_beaten': 0
        }
        self.decision_history = []


class AdaptiveBalatroPolicy(EnhancedBalatroPolicy):
    """
    Adaptive version that learns from experience and adjusts strategies.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adaptation_rate = 0.1
        self.strategy_success_rates = {}
        self.recent_outcomes = []
        self.max_recent_outcomes = 50
    
    def select_action(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        """
        Select action with adaptive learning.
        """
        # Get base action from enhanced policy
        action = super().select_action(state, info)
        
        # Record decision context for learning
        decision_context = {
            'action': action,
            'state_summary': self._summarize_state(state),
            'strategy': self._analyze_game_state(state, info)['recommended_strategy'],
            'timestamp': len(self.decision_history)
        }
        
        self.decision_history.append(decision_context)
        
        return action
    
    def update_performance(self, reward: float, info: Dict[str, Any]):
        """
        Update performance with adaptive learning.
        """
        super().update_performance(reward, info)
        
        # Record outcome for recent decisions
        self.recent_outcomes.append(reward)
        if len(self.recent_outcomes) > self.max_recent_outcomes:
            self.recent_outcomes.pop(0)
        
        # Update strategy success rates
        if self.decision_history:
            recent_decision = self.decision_history[-1]
            strategy = recent_decision['strategy']
            
            if strategy not in self.strategy_success_rates:
                self.strategy_success_rates[strategy] = {'total': 0, 'success': 0}
            
            self.strategy_success_rates[strategy]['total'] += 1
            if reward > 0:
                self.strategy_success_rates[strategy]['success'] += 1
        
        # Adapt parameters based on recent performance
        self._adapt_parameters()
    
    def _adapt_parameters(self):
        """
        Adapt policy parameters based on recent performance.
        """
        if len(self.recent_outcomes) < 10:
            return
        
        avg_recent_reward = np.mean(self.recent_outcomes[-10:])
        
        # Adapt aggression based on success
        if avg_recent_reward > 5:  # Doing well
            self.aggression = min(1.0, self.aggression + self.adaptation_rate * 0.1)
        elif avg_recent_reward < 0:  # Doing poorly
            self.aggression = max(0.0, self.aggression - self.adaptation_rate * 0.1)
        
        # Adapt risk tolerance
        success_rate = len([r for r in self.recent_outcomes[-20:] if r > 0]) / min(20, len(self.recent_outcomes))
        if success_rate > 0.7:
            self.risk_tolerance = min(1.0, self.risk_tolerance + self.adaptation_rate * 0.05)
        elif success_rate < 0.3:
            self.risk_tolerance = max(0.0, self.risk_tolerance - self.adaptation_rate * 0.05)
    
    def _summarize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a compact summary of game state for learning.
        """
        return {
            'ante': state.get('ante', 1),
            'money': state.get('money', 0),
            'hands_left': state.get('hands_left', 0),
            'discards': state.get('discards', 0),
            'num_jokers': len(state.get('jokers', [])),
            'hand_size': len(state.get('hand', [])),
            'selected_size': len(state.get('selected_cards', []))
        }
    
    def get_policy_info(self) -> Dict[str, Any]:
        """
        Get policy info including adaptive learning metrics.
        """
        info = super().get_policy_info()
        info.update({
            'policy_subtype': 'adaptive',
            'adaptation_rate': self.adaptation_rate,
            'strategy_success_rates': self.strategy_success_rates.copy(),
            'recent_performance': np.mean(self.recent_outcomes) if self.recent_outcomes else 0.0
        })
        return info


# Factory function for creating enhanced policies
def create_enhanced_policy(policy_type: str = "enhanced", **kwargs) -> EnhancedBalatroPolicy:
    """
    Factory function to create enhanced Balatro policies.
    
    Args:
        policy_type: Type of enhanced policy ('enhanced', 'adaptive')
        **kwargs: Additional arguments passed to policy constructor
        
    Returns:
        Enhanced Balatro policy instance
    """
    if policy_type == "enhanced":
        return EnhancedBalatroPolicy(**kwargs)
    elif policy_type == "adaptive":
        return AdaptiveBalatroPolicy(**kwargs)
    else:
        raise ValueError(f"Unknown enhanced policy type: {policy_type}")


if __name__ == "__main__":
    # Example usage and testing
    import json
    
    print("Testing Enhanced Balatro Policy:")
    
    # Test enhanced policy
    policy = EnhancedBalatroPolicy(aggression=0.7, economic_focus=0.3, risk_tolerance=0.6)
    
    # Mock game state for testing
    test_state = {
        'hand': [
            {'rank': 'A', 'suit': '♠'}, {'rank': 'A', 'suit': '♥'},
            {'rank': 'K', 'suit': '♦'}, {'rank': 'Q', 'suit': '♣'},
            {'rank': 'J', 'suit': '♠'}, {'rank': '10', 'suit': '♥'},
            {'rank': '9', 'suit': '♦'}, {'rank': '8', 'suit': '♣'}
        ],
        'selected_cards': [0, 1],  # Two Aces selected
        'money': 15,
        'chips': 0,
        'mult': 1,
        'ante': 3,
        'discards': 2,
        'hands_left': 3,
        'jokers': [
            {'name': 'Greedy Joker', 'mult': 4, 'chips': 0, 'effect': '+4 Mult'},
            {'name': 'Lusty Joker', 'mult': 0, 'chips': 20, 'effect': '+20 Chips'}
        ],
        'phase': 'play'
    }
    
    test_info = {
        'phase': 'play',
        'chips_needed': 600,
        'blind_name': 'The Wall',
        'legal_actions': list(range(10))
    }
    
    # Test action selection
    action = policy.select_action(test_state, test_info)
    print(f"Selected action: {action}")
    
    # Test adaptive policy
    print("\nTesting Adaptive Balatro Policy:")
    adaptive_policy = AdaptiveBalatroPolicy(aggression=0.5, risk_tolerance=0.4)
    
    # Simulate some outcomes
    for i in range(5):
        action = adaptive_policy.select_action(test_state, test_info)
        reward = np.random.uniform(-2, 10)  # Random reward
        adaptive_policy.update_performance(reward, {'hand_won': reward > 0})
        print(f"Iteration {i+1}: Action={action}, Reward={reward:.1f}")
    
    # Show policy information
    print(f"\nEnhanced Policy Info:")
    print(json.dumps(policy.get_policy_info(), indent=2))
    
    print(f"\nAdaptive Policy Info:")
    print(json.dumps(adaptive_policy.get_policy_info(), indent=2))
