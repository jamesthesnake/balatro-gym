"""Reward shaping utilities for the *minimal* Balatro hand-selection env.

▪ Only the hooks actually used by the current env are *active*.
▪ All original full-game helpers are preserved but commented out so you can
  re-enable them when you expand scope (shops, blinds, antes, etc.).

Usage (inside env.step):
    reward += self.reward_shaper.on_play(self)
    ...
    reward += self.reward_shaper.on_discard(self)
    ...
    final = self.reward_shaper.on_end_hand(base_score, joker_hit, turns_used)

"""

from __future__ import annotations

class RewardShaper:
    """Shapes rewards to encourage good play patterns in the *mini* env."""

    # ------------------------------------------------------------------ init
    def __init__(self):
        # Minimal weights that matter **today**
        self.weights = {
            "hand_score": 0.01,     # scale raw chips×mult so PPO gradients sane
            "joker_synergy": 2.0,   # bonus if active joker multiplies score
            "efficiency": 1.0,      # finishing in <5 plays
            "reward_discard": 0.05, # tiny reward to explore discard loops
            "penalty_illegal": -1.0 # mask leak safeguard
        }

    # ------------------------------------------------------- active mini hooks
    def on_play(self, env) -> float:
        """Optional shaping each legal PLAY_CARD_k. Currently neutral."""
        return 0.0

    def on_discard(self, env) -> float:
        """Called right after DISCARD_HAND succeeds."""
        return self.weights["reward_discard"]

    def on_end_hand(self, base_score: int, joker_hit: bool, turns_used: int) -> float:
        """Final reward when episode terminates.

        Args
        ----
        base_score : int
            Raw chips×mult from `score_hand` (before joker mods).
        joker_hit : bool
            True if any active joker multiplied the score.
        turns_used : int
            Number of PLAY_CARD actions the agent took (1–5).
        """
        r = base_score * self.weights["hand_score"]
        if joker_hit:
            r += self.weights["joker_synergy"]
        r += self.weights["efficiency"] * max(0, 5 - turns_used) / 4
        return r

    # ---------------------------- utilities for unexpected illegal behaviour
    def illegal(self) -> float:
        return self.weights["penalty_illegal"]

    # ------------------------------------------------------------------------
    # Everything below is *out of current scope* but kept for future expansion.
    # Uncomment as you grow the environment (blinds, shop, antes, vouchers …)
    # ------------------------------------------------------------------------

    """
    # Original hand-level reward
    def calculate_hand_reward(self, hand_score: int, current_score: int,
                               target_score: int) -> float:
        reward = 0.0
        reward += hand_score * self.weights.get("hand_score", 0.01)
        progress = current_score / target_score
        if progress > 0.5:
            reward += 5.0
        if progress > 0.8:
            reward += 10.0
        return reward

    def calculate_blind_reward(self, final_score: int, target_score: int,
                                ante: int) -> float:
        reward = 0.0
        if final_score >= target_score:
            reward += self.weights.get("blind_completion", 100)
            overkill = (final_score - target_score) / target_score
            reward += min(overkill * 20, 50)
            reward *= (1 + ante * 0.1)
        else:
            reward -= 50
            completion = final_score / target_score
            reward += completion * 20
        return reward

    def calculate_shop_reward(self, action: str, cost: int,
                              money_before: int, money_after: int) -> float:
        reward = 0.0
        if money_after < 0:
            return -100.0
        purchase_values = {
            "joker": 5.0,
            "tarot": 2.0,
            "planet": 2.0,
            "spectral": 3.0,
            "voucher": 10.0,
        }
        for k, v in purchase_values.items():
            if k in action.lower():
                reward += v
                break
        if money_after > 10:
            reward += self.weights.get("money_saved", 0.1) * money_after
        return reward

    def calculate_episode_bonus(self, final_ante: int, total_money: int,
                                victory: bool) -> float:
        reward = 0.0
        if victory:
            reward += 1000.0
        reward += (final_ante - 1) * self.weights.get("ante_progression", 200)
        if total_money > 50:
            reward += total_money * self.weights.get("money_saved", 0.1)
        return reward

    # --- full-game hooks -----------------------------------------------------
    def on_skip_blind(self, env_state) -> float:
        return 0.0

    def on_use_consumable(self, consumable, env_state) -> float:
        return 1.0

    def on_buy_joker(self, joker, env_state) -> float:
        return self.calculate_shop_reward("buy_joker", joker.cost,
                                           env_state.money + joker.cost,
                                           env_state.money)

    def on_buy_pack(self, pack_type: str, env_state) -> float:
        cost = 4  # base demo cost
        return self.calculate_shop_reward(f"buy_pack_{pack_type}", cost,
                                           env_state.money + cost,
                                           env_state.money)

    def on_buy_voucher(self, voucher, env_state) -> float:
        return self.calculate_shop_reward("buy_voucher", voucher.cost,
                                           env_state.money + voucher.cost,
                                           env_state.money)

    def on_no_op(self, env_state) -> float:
        return self.weights.get("penalty_no_op", -0.1)
    """

