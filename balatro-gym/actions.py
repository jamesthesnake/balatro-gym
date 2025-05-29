from enum import IntEnum, auto


class Action(IntEnum):
    # ─── Card play ────────────────────────────────────────────────────────────────
    PLAY_CARD_0 = 0
    PLAY_CARD_1 = 1
    PLAY_CARD_2 = 2
    PLAY_CARD_3 = 3
    PLAY_CARD_4 = 4

    # ─── Hand-level control ───────────────────────────────────────────────────────
    DISCARD_HAND   = 5          # Burn current hand & draw new one
    END_HAND       = 6          # “Play” without discarding
    SKIP_BLIND     = 7          # Pay chips to skip current blind

    # ─── Consumables ─────────────────────────────────────────────────────────────
    USE_CONSUMABLE = 8          # Use the single consumable in inventory

    # ─── Shop buys (index-addressed to keep things deterministic) ────────────────
    BUY_JOKER_SLOT_0 = 9
    BUY_JOKER_SLOT_1 = 10
    BUY_JOKER_SLOT_2 = 11
    BUY_JOKER_SLOT_3 = 12
    BUY_JOKER_SLOT_4 = 13

    BUY_PACK_TAROT    = 14
    BUY_PACK_PLANET   = 15
    BUY_PACK_SPECTRAL = 16
    BUY_VOUCHER       = 17

    # ─── Meta control ────────────────────────────────────────────────────────────
    NO_OP = 18                 # Safe fallback if the mask/query fails

    @classmethod
    def count(cls) -> int:
        """Convenience for gym.spaces.Discrete."""
        return len(cls)
