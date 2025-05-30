"""Simplified Balatro environment compatible with SB3‑Contrib MaskablePPO.

Scope reduction:
    • Focuses on a **single blind attempt**: draw 7 cards, agent selects
      exactly 5 to play (via sequential PLAY_CARD_* actions) or ends early.
    • Shop, consumable, voucher and blind‑skipping mechanics are **commented
      out** but kept as placeholders so you can incrementally re‑enable them.
    • Uses `RewardShaper` hook points for future shaping, but only
      `on_play()` and `on_end_hand()` are called for now.

External dependencies you already have in the repo:
    • `Deck` + `Card` primitives in `balatro_gym.cards`
    • `HandEvaluator.evaluate()` returning `(hand_type, scoring_cards)`
    • `RewardShaper` with `.on_play(card, env)` and `.on_end_hand(env)`

This file will **replace** your previous heavy `BalatroEnv` implementation.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Tuple, Any
from enum import IntEnum

# -----------------------------------------------------------------------------
# Trimmed Action enum (card play + hand control only)
# -----------------------------------------------------------------------------
class Action(IntEnum):
    PLAY_CARD_0 = 0
    PLAY_CARD_1 = 1
    PLAY_CARD_2 = 2
    PLAY_CARD_3 = 3
    PLAY_CARD_4 = 4
    DISCARD_HAND = 5   # Currently disabled by mask once a card is played
    END_HAND     = 6

    # Future scope (shops, vouchers, etc.) – keep IDs reserved
    # SKIP_BLIND     = 7
    # USE_CONSUMABLE = 8
    # BUY_JOKER_SLOT_0 = 9
    # BUY_JOKER_SLOT_1 = 10
    # BUY_JOKER_SLOT_2 = 11
    # BUY_JOKER_SLOT_3 = 12
    # BUY_JOKER_SLOT_4 = 13
    # BUY_PACK_TAROT    = 14
    # BUY_PACK_PLANET   = 15
    # BUY_PACK_SPECTRAL = 16
    # BUY_VOUCHER       = 17
    # NO_OP             = 18

    @classmethod
    def count(cls) -> int:
        return len(cls)

# -----------------------------------------------------------------------------
# Imports that exist elsewhere in the repo
# -----------------------------------------------------------------------------
from balatro_gym.cards import Deck, Card            # type: ignore
from balatro_gym.hand_eval import HandEvaluator     # type: ignore
from balatro_gym.reward_shaper import RewardShaper  # type: ignore
# StateFormatter kept for future LLM context dumps; unused in env logic
from balatro_gym.state_formatter import StateFormatter  # type: ignore

# -----------------------------------------------------------------------------
# Minimal Balatro environment
# -----------------------------------------------------------------------------
class BalatroEnv(gym.Env):
    """One‑blind Balatro hand‑selection environment."""

    metadata = {"render_modes": []}

    def __init__(self, *, seed: int | None = None, hand_size: int = 7):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.hand_size = hand_size

        # Core game state ------------------------------------------------------
        self.money = 4              # unused until shop added
        self.target_score = 0       # can be set externally for shaping
        self.current_score = 0
        self.hands_remaining = 1
        self.discards_remaining = 1

        # Deck & hand ----------------------------------------------------------
        self.deck = Deck(self.rng)
        self.hand: List[Card] = []
        self.played_idx: List[int] = []

        # Helpers --------------------------------------------------------------
        self.reward_shaper = RewardShaper()
        self.state_formatter = StateFormatter()

        # Gym spaces -----------------------------------------------------------
        self.action_space = spaces.Discrete(Action.count())
        self.observation_space = spaces.Dict(
            {
                "hand": spaces.Box(0, 12, shape=(hand_size, 2), dtype=np.int8),
                "played_mask": spaces.MultiBinary(hand_size),
                "action_mask": spaces.MultiBinary(Action.count()),
            }
        )

    # ---------------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self.deck.reset()
        self.hand = self.deck.draw(self.hand_size)
        self.played_idx.clear()
        self.current_score = 0
        return self._get_obs(), {}

    def step(self, action_idx: int):
        action = Action(action_idx)
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        # ---------------- Execute action ------------------
        if action.name.startswith("PLAY_CARD"):
            idx = action.value - Action.PLAY_CARD_0
            if idx in self.played_idx:
                reward = -1.0  # illegal
                terminated = True
            else:
                self.played_idx.append(idx)
                card = self.hand[idx]
                reward += self.reward_shaper.on_play(card, self)
                if len(self.played_idx) == 5:
                    reward += self._score_hand()
                    terminated = True
        elif action == Action.DISCARD_HAND:
            if self.played_idx:  # can’t discard mid‑selection
                reward = -1.0
                terminated = True
            else:
                self.hand = self.deck.draw(self.hand_size)
                self.discards_remaining -= 1
        elif action == Action.END_HAND:
            reward += self._score_hand()
            terminated = True
        else:
            raise ValueError(f"Unhandled action {action}")

        return self._get_obs(), reward, terminated, truncated, info

    # ---------------------------------------------------------------------
    # Scoring helpers
    # ---------------------------------------------------------------------
    def _score_hand(self) -> float:
        selected = [self.hand[i] for i in self.played_idx]
        if len(selected) < 2:
            # Too few cards = dead hand
            return -0.5
        hand_type, scoring_cards = HandEvaluator.evaluate(selected)
        chips = hand_type.get_base_chips()
        chips += sum(c.get_chip_value() for c in scoring_cards)
        mult = hand_type.get_base_mult()
        score = chips * mult
        self.current_score += score
        return self.reward_shaper.on_end_hand(self) + score / 10_000.0

    # ---------------------------------------------------------------------
    # Observation & masks
    # ---------------------------------------------------------------------
    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(Action.count(), dtype=np.int8)

        # Card plays -------------------------------------------------------
        for i in range(5):  # PLAY_CARD_0‑4
            if i < len(self.hand) and i not in self.played_idx:
                mask[Action.PLAY_CARD_0 + i] = 1

        # Discard / End hand ----------------------------------------------
        if not self.played_idx and self.discards_remaining > 0:
            mask[Action.DISCARD_HAND] = 1
        mask[Action.END_HAND] = 1  # always allow ending early

        return mask

    def _get_obs(self):
        hand_arr = np.array([c.to_int_pair() for c in self.hand], dtype=np.int8)
        played_mask = np.zeros(self.hand_size, dtype=np.int8)
        played_mask[self.played_idx] = 1
        return {
            "hand": hand_arr,
            "played_mask": played_mask,
            "action_mask": self._action_mask(),
        }

# -----------------------------------------------------------------------------
# Factory for SB3
# -----------------------------------------------------------------------------

def make_env(seed: int | None = None):
    return BalatroEnv(seed=seed)

# -----------------------------------------------------------------------------
# Quick manual test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    env = BalatroEnv(seed=0)
    obs, _ = env.reset()
    done, ep_ret = False, 0
    while not done:
        legal = np.flatnonzero(obs["action_mask"])
        act = int(env.rng.choice(legal))
        obs, rew, done, _, _ = env.step(act)
        ep_ret += rew
    print(f"Random episode return: {ep_ret:.3f}")

