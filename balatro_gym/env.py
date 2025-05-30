import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from typing import Dict, Tuple, Any

from .actions import Action
from .state_formatter import StateFormatter
from .reward_shaper import RewardShaper
from .cards import Deck, HandEvaluator
from .jokers import JokerShop
from .consumables import ConsumableShop
from .vouchers import VoucherShop

class BalatroEnv(gym.Env):
    def __init__(self, *, render_mode: str | None = None, **kwargs):
        super().__init__()
        self.render_mode = render_mode
        # -- core game state --
        self.money = 0
        self.current_ante = 1
        self.current_blind = "small"
        self.target_score = 0
        self.current_score = 0
        self.hands_remaining = 4
        self.discards_remaining = 3
        # -- game parameters --
        self.base_hands = 4
        self.base_discards = 3
        self.hand_size = 8
        self.joker_slots = 5
        self.consumable_slots = 2
        self.shop_slots = 2
        self.reroll_cost = 5
        self.interest_cap = 5
        # -- decks and inventories --
        self.deck = Deck()
        self.hand: list = []
        self.jokers: list = []
        self.consumables: list = []
        # -- shops --
        self.in_shop = False
        self.shop_jokers: list = []
        self.shop_packs: dict = {}
        self.shop_voucher = None
        # -- effect state --
        self.voucher_effects: dict = {}
        # -- history for LLM context --
        self.action_history = deque(maxlen=20)
        self.hand_history = deque(maxlen=10)
        # -- helpers --
        self.state_formatter = StateFormatter()
        self.reward_shaper = RewardShaper()
        # -- spaces --
        self.observation_space = spaces.Dict({
            "money": spaces.Box(0, 9999, (1,), np.int32),
            "ante": spaces.Box(1, 20, (1,), np.int32),
            "target_score": spaces.Box(0, 999999, (1,), np.int32),
            "current_score": spaces.Box(0, 999999, (1,), np.int32),
            "hands_remaining": spaces.Box(0, 10, (1,), np.int32),
            "discards_remaining": spaces.Box(0, 10, (1,), np.int32),
            "in_shop": spaces.Box(0, 1, (1,), np.int32),
            "action_mask": spaces.MultiBinary(Action.count()),
        })
        self.action_space = spaces.Discrete(Action.count())

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Reset state
        self.money = 4
        self.current_ante = 1
        self.current_blind = "small"
        self.target_score = 300
        self.current_score = 0
        self.base_hands = 4
        self.base_discards = 3
        self.hands_remaining = self.base_hands
        self.discards_remaining = self.base_discards
        # Reset deck and hand
        self.deck.reset()
        self.hand = self.deck.draw(self.hand_size)
        self.jokers = []
        self.consumables = []
        # Reset shop
        self.in_shop = False
        self.shop_jokers = []
        self.shop_packs = {}
        self.shop_voucher = None
        self.voucher_effects = {}
        # Reapply vouchers if any
        self._apply_voucher_effects()
        self.action_history.clear()
        self.hand_history.clear()
        obs = self._get_obs()
        info = {"ante": self.current_ante, "blind": self.current_blind}
        return obs, info

    def step(self, action_idx: int):
        action = Action(action_idx)
        self.action_history.append({
            "action": action.name,
            "ante": self.current_ante,
            "money": self.money
        })
        reward, terminated, truncated, info = self._apply_action(action)
        obs = self._get_obs()
        if self.render_mode == "human":
            self._render_human()
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "money": np.array([self.money], np.int32),
            "ante": np.array([self.current_ante], np.int32),
            "target_score": np.array([self.target_score], np.int32),
            "current_score": np.array([self.current_score], np.int32),
            "hands_remaining": np.array([self.hands_remaining], np.int32),
            "discards_remaining": np.array([self.discards_remaining], np.int32),
            "in_shop": np.array([int(self.in_shop)], np.int32),
            "action_mask": self.action_mask(),
        }

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(Action.count(), np.int8)
        # -- fill in legal actions as before --
        # (Same as your existing implementation)
        return mask

    def _apply_action(self, action: Action) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """
        Apply a chosen action, update game state, and return
        (reward, terminated, truncated, info).
        """
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        # Example card-play logic
        if action.name.startswith("PLAY_CARD_"):
            idx = action.value - Action.PLAY_CARD_0.value
            
            # --- Important Note on Gameplay Logic for PLAY_CARD_X ---
            # Currently, one card is popped, and then the *remaining* hand is evaluated.
            # In Balatro, you typically select 1-5 cards, and *those selected cards* are evaluated.
            # The logic below fixes the TypeError based on the current structure,
            # but you may want to revisit how 'played_hand_cards' are determined.
            # For example, if PLAY_CARD_X means "play card at hand[idx] AND other selected cards",
            # you'd need a selection mechanism.
            # If it means "these are the 1-5 cards selected by the agent to play this turn",
            # then 'idx' might not be used, or PLAY_CARD_X actions might represent full hand plays.
            
            if idx < len(self.hand): # Ensure index is valid before popping
                played_card_for_reward_shaping = self.hand.pop(idx) # Card for reward shaper
                
                # For evaluation, Balatro evaluates the specific cards played (1-5 cards).
                # Currently, your code evaluates self.hand *after* popping one.
                # Let's assume for now 'self.hand' (after pop) are the cards to evaluate.
                # If PLAY_CARD_X action implies a specific set of cards to be played,
                # those should be passed to HandEvaluator.evaluate().
                # For example, if your intent was to play only the popped card (which is unusual for Balatro scoring):
                # played_cards_for_evaluation = [played_card_for_reward_shaping]
                # Or if it's a pre-selected hand that 'idx' refers to within that selection:
                # # (This would require a card selection mechanism first)

                # Using current structure: evaluating the remaining self.hand
                cards_to_evaluate = self.hand 
                # If your game logic means PLAY_CARD_X implies playing a specific *set* of cards
                # (e.g., cards previously selected by the agent), then `cards_to_evaluate`
                # should be that set of cards, not just `self.hand`.

                if cards_to_evaluate: # Only evaluate if there are cards
                    hand_type, scoring_cards = HandEvaluator.evaluate(cards_to_evaluate)
                    
                    # Calculate score
                    current_hand_chips = hand_type.get_base_chips()
                    for card_in_scoring_hand in scoring_cards:
                        current_hand_chips += card_in_scoring_hand.get_chip_value()
                    
                    current_hand_mult = hand_type.get_base_mult()
                    
                    calculated_score = current_hand_chips * current_hand_mult
                    self.current_score += calculated_score
                else:
                    # No cards to evaluate, or handle as per your game rules (e.g., score 0)
                    calculated_score = 0 
                    # self.current_score += 0 (no change)

                self.hands_remaining -= 1
                reward += self.reward_shaper.on_play(played_card_for_reward_shaping, self)
            else:
                # Invalid action if idx is out of bounds (e.g., trying to play a card from an empty hand slot)
                # This should ideally be caught by the action_mask, but good to have a fallback.
                print(f"Warning: Tried to play card at invalid index {idx} from hand.")
                # Handle appropriately, e.g., small penalty or no operation.
                pass

        elif action == Action.DISCARD_HAND:
            self.hand = self.deck.draw(self.hand_size)
            self.discards_remaining -= 1
            reward += self.reward_shaper.on_discard(self)

        elif action == Action.END_HAND:
            # TODO: finalize hand
            reward += self.reward_shaper.on_end_hand(self)
            self.hands_remaining = 0

        elif action == Action.SKIP_BLIND:
            cost = self._skip_cost()
            self.money -= cost
            reward -= cost * 0.1

        elif action == Action.USE_CONSUMABLE:
            consumable = self.consumables.pop(0)
            reward += self.reward_shaper.on_use_consumable(consumable, self)

        elif action.name.startswith("BUY_JOKER_SLOT_"):
            idx = action.value - Action.BUY_JOKER_SLOT_0.value
            joker = self.shop_jokers[idx]
            self.money -= joker.cost
            self.jokers.append(joker)
            reward += self.reward_shaper.on_buy_joker(joker, self)

        elif action in (Action.BUY_PACK_TAROT, Action.BUY_PACK_PLANET, Action.BUY_PACK_SPECTRAL):
            pack_type = action.name.split("_")[-1].lower()
            cost = 4 * (1 - self.voucher_effects.get("shop_discount", 0))
            self.money -= cost
            new_cards = self.deck.draw_pack(pack_type)  # assume method exists
            self.hand.extend(new_cards)
            reward += self.reward_shaper.on_buy_pack(pack_type, self)

        elif action == Action.BUY_VOUCHER:
            voucher = self.shop_voucher
            self.money -= voucher.cost
            effects = self.voucher_shop.apply(voucher)
            self.voucher_effects.update(effects)
            reward += self.reward_shaper.on_buy_voucher(voucher, self)

        elif action == Action.NO_OP:
            reward += self.reward_shaper.on_no_op(self)

        else:
            raise ValueError(f"Unknown action: {action}")

        # Termination check
        if self.hands_remaining <= 0 or self.discards_remaining < 0:
            terminated = True
            info['reason'] = 'no hands or discards'

        # TODO: manage shop entry/exit
        # self._maybe_open_shop()

        return reward, terminated, truncated, info
    def _apply_voucher_effects(self):
    	"""
    	Apply any active voucher effects to modify game state.
    	Right now this is a stub to prevent crashes.
	 """
    	pass
    def _skip_cost(self):
        """
        Helper method to determine the cost of skipping a blind.
        Implement your logic here, e.g., fixed cost, or depends on ante.
        """
        # Example:
        # if self.current_blind == "small":
        #     return 5 + self.current_ante
        # elif self.current_blind == "big":
        #     return 10 + self.current_ante
        return 5 # Placeholder: Define actual skip cost logic

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(Action.count(), dtype=np.int8)

        # --- META CONTROL ---
        # NO_OP is often a safe fallback.
        # Consider if it should always be available or only in specific circumstances.
        mask[Action.NO_OP.value] = 1

        # --- SHOP ACTIONS ---
        if self.in_shop:
            # BUY_JOKER_SLOT_0 to BUY_JOKER_SLOT_4
            for i in range(5): # Corresponds to BUY_JOKER_SLOT_0 through BUY_JOKER_SLOT_4
                action_val = Action.BUY_JOKER_SLOT_0.value + i
                if i < len(self.shop_jokers): # Check if a joker exists in this shop slot
                    joker_in_shop = self.shop_jokers[i]
                    # Ensure joker_in_shop has a 'cost' attribute
                    if hasattr(joker_in_shop, 'cost') and self.money >= joker_in_shop.cost and len(self.jokers) < self.joker_slots:
                        mask[action_val] = 1
            
            # BUY_PACK_TAROT, BUY_PACK_PLANET, BUY_PACK_SPECTRAL
            # Assuming pack_cost is consistent for these types as in _apply_action
            pack_base_cost = 4 # From your _apply_action
            effective_pack_cost = pack_base_cost * (1 - self.voucher_effects.get("shop_discount", 0))

            # Check if packs are available in the shop (e.g., self.shop_packs might store quantity or availability)
            # For simplicity here, we'll just check money. You might need more detailed pack availability logic.
            if 'tarot' in self.shop_packs and self.shop_packs['tarot'] > 0: # Example: check if pack exists and quantity > 0
                if self.money >= effective_pack_cost:
                    mask[Action.BUY_PACK_TAROT.value] = 1
            
            if 'planet' in self.shop_packs and self.shop_packs['planet'] > 0:
                if self.money >= effective_pack_cost:
                    mask[Action.BUY_PACK_PLANET.value] = 1

            if 'spectral' in self.shop_packs and self.shop_packs['spectral'] > 0:
                if self.money >= effective_pack_cost:
                    mask[Action.BUY_PACK_SPECTRAL.value] = 1

            # BUY_VOUCHER
            if self.shop_voucher is not None:
                # Ensure self.shop_voucher has a 'cost' attribute
                if hasattr(self.shop_voucher, 'cost') and self.money >= self.shop_voucher.cost:
                    # Also check if voucher is not already owned if it's unique
                    mask[Action.BUY_VOUCHER.value] = 1
            
            # MISSING ACTIONS: REROLL_SHOP and LEAVE_SHOP
            # Your Action enum doesn't have these. This might make it hard to
            # leave the shop or refresh it. Consider how game flow handles this.
            # If NO_OP is meant to leave the shop when in shop and no other action is taken,
            # that logic would be part of your _apply_action for NO_OP when self.in_shop is true.

        # --- ROUND ACTIONS (NOT IN SHOP) ---
        else: # not self.in_shop
            # PLAY_CARD_0 to PLAY_CARD_4
            # These actions imply playing the card at the given hand index.
            # The actual game of Balatro involves selecting 1-5 cards and playing a poker hand.
            # This implementation is based on your current _apply_action.
            if self.hands_remaining > 0:
                for i in range(5): # PLAY_CARD_0 to PLAY_CARD_4
                    action_val = Action.PLAY_CARD_0.value + i
                    if i < len(self.hand): # Check if card index is valid for current hand
                        mask[action_val] = 1
            
            # DISCARD_HAND
            if self.discards_remaining > 0 and len(self.hand) > 0 : # Can only discard if you have cards and discards left
                mask[Action.DISCARD_HAND.value] = 1

            # END_HAND
            # Based on _apply_action, this seems to be for ending the current attempt at the blind.
            # It might be available if you've played at least one card or made some move,
            # or simply if hands_remaining > 0.
            if self.hands_remaining > 0: # Or some other condition like cards have been selected
                 mask[Action.END_HAND.value] = 1
            
            # SKIP_BLIND
            cost_to_skip = self._skip_cost()
            if self.money >= cost_to_skip:
                mask[Action.SKIP_BLIND.value] = 1
            
            # USE_CONSUMABLE
            if len(self.consumables) > 0: # Assuming self.consumables is a list of available consumables
                 mask[Action.USE_CONSUMABLE.value] = 1
            
            # MISSING ACTION: GO_TO_SHOP
            # How does self.in_shop become True? This transition needs to be handled,
            # possibly automatically in _apply_action after a blind is completed (win or lose).

        return mask

    def _render_human(self):
        """
        Simple text-based rendering of the current state.
        """
        print("=== Balatro State ===")
        print(f"Money: {self.money} | Score: {self.current_score}/{self.target_score}")
        print(f"Hand ({len(self.hand)}/{self.hand_size}): {self.hand}")
        print(f"Jokers: {self.jokers} | Consumables: {self.consumables}")
        if self.in_shop:
            print("--- Shop View ---")
            print(f"Jokers: {self.shop_jokers}")
            print(f"Packs: {list(self.shop_packs.keys())}")
            print(f"Voucher: {self.shop_voucher}")
        print(f"Hands left: {self.hands_remaining} | Discards left: {self.discards_remaining}")
        print("====================")
