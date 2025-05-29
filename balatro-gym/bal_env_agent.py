import gymnasium as gym
from gymnasium import spaces
from .actions import Action

class BalatroEnv(gym.Env):

    def __init__(self, *, render_mode: str | None = None, **kwargs):
        super().__init__()
        self.render_mode = render_mode
        # observation_space = ...            # ← leave whatever you had
        self.action_space = spaces.Discrete(Action.count())

    # --------------------------------------------------------------------- public
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        ...
        return obs, info | {}

    def step(self, action_idx: int):
        action = Action(action_idx)
        reward, terminated, truncated, info = self._apply_action(action)
        obs = self._get_obs()
        if self.render_mode == "human":
            self._render_human()
        return obs, reward, terminated, truncated, info

    def action_mask(self):
        """
        Returns a 0/1 np.ndarray[len(Action)] mask where 1 == legal.
        Call it from your agent as `env.action_mask()`.
        """
        mask = np.zeros(Action.count(), dtype=np.int8)
        # --- example legality checks (you’ll flesh these out) -------------------
        if self._can_play_card(0):   mask[Action.PLAY_CARD_0] = 1
        ...
        if self._shop_has_tarot():   mask[Action.BUY_PACK_TAROT] = 1
        mask[Action.NO_OP] = 1  # always legal
        return mask

    # ------------------------------------------------------------------ internal
    def _apply_action(self, action: Action):
        """Route each verb to its handler. Returns (reward, terminated, truncated, info)."""
        match action:
            case Action.PLAY_CARD_0 | Action.PLAY_CARD_1 | Action.PLAY_CARD_2 \
                 | Action.PLAY_CARD_3 | Action.PLAY_CARD_4:
                return self._act_play_card(action.value)
            case Action.DISCARD_HAND:
                return self._act_discard()
            case Action.END_HAND:
                return self._act_end_hand()
            case Action.SKIP_BLIND:
                return self._act_skip_blind()
            case Action.USE_CONSUMABLE:
                return self._act_use_consumable()
            case Action.BUY_JOKER_SLOT_0 | Action.BUY_JOKER_SLOT_1 \
                 | Action.BUY_JOKER_SLOT_2 | Action.BUY_JOKER_SLOT_3 | Action.BUY_JOKER_SLOT_4:
                return self._act_buy_joker(slot=action.value - Action.BUY_JOKER_SLOT_0)
            case Action.BUY_PACK_TAROT:
                return self._act_buy_pack("tarot")
            case Action.BUY_PACK_PLANET:
                return self._act_buy_pack("planet")
            case Action.BUY_PACK_SPECTRAL:
                return self._act_buy_pack("spectral")
            case Action.BUY_VOUCHER:
                return self._act_buy_voucher()
            case Action.NO_OP:
                return 0.0, False, False, {"reason": "noop"}
            case _:
                raise ValueError(f"Unhandled action {action}")

    # ------------- individual verb handlers (stubbed – implement game logic) ---
    def _act_play_card(self, hand_idx: int): ...
    def _act_discard(self): ...
    def _act_end_hand(self): ...
    def _act_skip_blind(self): ...
    def _act_use_consumable(self): ...
    def _act_buy_joker(self, slot: int): ...
    def _act_buy_pack(self, pack_type: str): ...
    def _act_buy_voucher(self): ...
