import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from typing import Dict, Tuple, Any, Optional, List

from .actions import Action
from .state_formatter import StateFormatter
from .reward_shaper import RewardShaper
from .cards import Card, Deck, HandEvaluator, HandType, Rank, Suit
from .jokers import Joker, JokerShop, JokerRarity
from .consumables import Consumable, ConsumableShop, ConsumableType
from .vouchers import Voucher, VoucherShop

class BalatroEnv(gym.Env):
    def __init__(self, *, render_mode: str | None = None, **kwargs):
        super().__init__()
        self.render_mode = render_mode
        
        # Core game state
        self.money = 0
        self.current_ante = 1
        self.current_blind = "small"  # small, big, boss
        self.target_score = 0
        self.current_score = 0
        self.hands_remaining = 4
        self.discards_remaining = 3
        
        # Game settings (modified by vouchers)
        self.base_hands = 4
        self.base_discards = 3
        self.hand_size = 8
        self.joker_slots = 5
        self.consumable_slots = 2
        self.shop_slots = 2  # Base joker slots in shop
        self.reroll_cost = 5
        self.interest_cap = 5
        
        # Cards and collections
        self.deck = Deck()
        self.hand = []  # Current hand of cards
        self.jokers = []  # Active jokers (max 5 by default)
        self.consumables = []  # Consumables (max 2 by default)
        self.hand_levels = {hand_type: 1 for hand_type in HandType}  # Planet card levels
        
        # Shop state
        self.in_shop = False
        self.shop_jokers = []
        self.shop_packs = {}
        self.shop_voucher = None
        self.shop_cards = []  # If voucher enables
        
        # Shop managers
        self.joker_shop = JokerShop()
        self.consumable_shop = ConsumableShop()
        self.voucher_shop = VoucherShop()
        
        # Voucher effects
        self.voucher_effects = {}
        
        # History tracking for LLM context
        self.action_history = deque(maxlen=20)
        self.hand_history = deque(maxlen=10)  # Recent hand scores
        self.last_consumable = None
        self.termination_reason = None
        
        # Helpers
        self.state_formatter = StateFormatter()
        self.reward_shaper = RewardShaper()
        
        # Gym spaces
        self.observation_space = spaces.Dict({
            "money": spaces.Box(0, 9999, shape=(1,), dtype=np.int32),
            "ante": spaces.Box(1, 20, shape=(1,), dtype=np.int32),
            "target_score": spaces.Box(0, 999999, shape=(1,), dtype=np.int32),
            "current_score": spaces.Box(0, 999999, shape=(1,), dtype=np.int32),
            "hands_remaining": spaces.Box(0, 10, shape=(1,), dtype=np.int32),
            "discards_remaining": spaces.Box(0, 10, shape=(1,), dtype=np.int32),
            "in_shop": spaces.Box(0, 1, shape=(1,), dtype=np.int32),
        })
        self.action_space = spaces.Discrete(Action.count())
    
    # --------------------------------------------------------------------- public
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        
        # Reset game state
        self.money = 4
        self.current_ante = 1
        self.current_blind = "small"
        self.target_score = 300
        self.current_score = 0
        
        # Reset game settings
        self.base_hands = 4
        self.base_discards = 3
        self.hand_size = 8
        self.joker_slots = 5
        self.consumable_slots = 2
        self.shop_slots = 2
        self.reroll_cost = 5
        self.interest_cap = 5
        
        # Reset deck and draw initial hand
        self.deck.reset()
        self.hand = self.deck.draw(self.hand_size)
        self.jokers = []
        self.consumables = []
        self.hand_levels = {hand_type: 1 for hand_type in HandType}
        
        # Reset shop
        self.in_shop = False
        self.shop_jokers = []
        self.shop_packs = {}
        self.shop_voucher = None
        self.shop_cards = []
        
        # Reset managers
        self.joker_shop = JokerShop()
        self.consumable_shop = ConsumableShop()
        self.voucher_shop = VoucherShop()
        self.voucher_effects = {}
        
        # Apply starting voucher effects if any
        self._apply_voucher_effects()
        
        # Set hands and discards for first blind
        self.hands_remaining = self.base_hands
        self.discards_remaining = self.base_discards
        
        # Clear history
        self.action_history.clear()
        self.hand_history.clear()
        self.last_consumable = None
        self.termination_reason = None
        
        obs = self._get_obs()
        info = {"ante": self.current_ante, "blind": self.current_blind}
        
        return obs, info
    
    def step(self, action_idx: int):
        action = Action(action_idx)
        
        # Record action in history
        self.action_history.append({
            "action": action.name,
            "action_idx": action_idx,
            "ante": self.current_ante,
            "blind": self.current_blind,
            "money": self.money
        })
        
        # Apply action
        reward, terminated, truncated, info = self._apply_action(action)
        
        # Get new observation
        obs = self._get_obs()
        
        # Render if needed
        if self.render_mode == "human":
            self._render_human()
        
        return obs, reward, terminated, truncated, info
    
    def action_mask(self) -> np.ndarray:
        """
        Returns a 0/1 np.ndarray[len(Action)] mask where 1 == legal.
        Call it from your agent as `env.action_mask()`.
        """
        mask = np.zeros(Action.count(), dtype=np.int8)
        
        # Card play actions
        for i in range(5):
            if i < len(self.hand) and not self.in_shop and self.hands_remaining > 0:
                mask[Action.PLAY_CARD_0 + i] = 1
        
        # Hand management
        if not self.in_shop and self.hands_remaining > 0:
            if self.discards_remaining > 0:
                mask[Action.DISCARD_HAND] = 1
            mask[Action.END_HAND] = 1
        
        # Skip blind (if have enough money)
        if not self.in_shop and self.money >= self._skip_cost():
            mask[Action.SKIP_BLIND] = 1
        
        # Consumable
        if self.consumables and not self.in_shop:
            mask[Action.USE_CONSUMABLE] = 1
        
        # Shop actions
        if self.in_shop:
            # Jokers
            for i, joker in enumerate(self.shop_jokers[:5]):
                if joker and self.money >= joker.cost and len(self.jokers) < self.joker_slots:
                    mask[Action.BUY_JOKER_SLOT_0 + i] = 1
            
            # Packs
            pack_cost = 4 * (1 - self.voucher_effects.get("shop_discount", 0))
            if self.shop_packs.get("tarot") and self.money >= pack_cost:
                mask[Action.BUY_PACK_TAROT] = 1
            if self.shop_packs.get("planet") and self.money >= pack_cost:
                mask[Action.BUY_PACK_PLANET] = 1
            if self.shop_packs.get("spectral") and self.money >= pack_cost:
                mask[Action.BUY_PACK_SPECTRAL] = 1
            
            # Voucher
            if self.shop_voucher and self.money >= self.shop_voucher.cost:
                mask[Action.BUY_VOUCHER] = 1
        
        # NO_OP is always legal
        mask[Action.NO_OP] = 1
        
        return mask
