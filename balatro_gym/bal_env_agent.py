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
            card = self.hand.pop(idx)
            # TODO: apply card effects
            score = HandEvaluator.evaluate(self.hand)
            self.current_score += score
            reward += self.reward_shaper.on_play(card, self)
            self.hands_remaining -= 1

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
