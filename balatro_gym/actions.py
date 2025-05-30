from enum import IntEnum, auto
from typing import Dict, List, Tuple

class Action(IntEnum):
    # ─── Card play ────────────────────────────────────────────────────────────────
    PLAY_CARD_0 = 0
    PLAY_CARD_1 = 1
    PLAY_CARD_2 = 2
    PLAY_CARD_3 = 3
    PLAY_CARD_4 = 4
    
    # ─── Hand-level control ───────────────────────────────────────────────────────
    DISCARD_HAND   = 5          # Burn current hand & draw new one
    END_HAND       = 6          # "Play" without discarding
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
    
    @classmethod
    def get_action_groups(cls) -> Dict[str, List[Tuple[int, str]]]:
        """Returns actions organized by category for better LLM understanding."""
        return {
            "Card Play": [
                (cls.PLAY_CARD_0, "Play card 0"),
                (cls.PLAY_CARD_1, "Play card 1"),
                (cls.PLAY_CARD_2, "Play card 2"),
                (cls.PLAY_CARD_3, "Play card 3"),
                (cls.PLAY_CARD_4, "Play card 4"),
            ],
            "Hand Management": [
                (cls.DISCARD_HAND, "Discard hand and draw new cards"),
                (cls.END_HAND, "End hand without playing"),
                (cls.SKIP_BLIND, "Skip blind (costs chips)"),
            ],
            "Consumables": [
                (cls.USE_CONSUMABLE, "Use consumable item"),
            ],
            "Shop - Jokers": [
                (cls.BUY_JOKER_SLOT_0, "Buy joker in slot 0"),
                (cls.BUY_JOKER_SLOT_1, "Buy joker in slot 1"),
                (cls.BUY_JOKER_SLOT_2, "Buy joker in slot 2"),
                (cls.BUY_JOKER_SLOT_3, "Buy joker in slot 3"),
                (cls.BUY_JOKER_SLOT_4, "Buy joker in slot 4"),
            ],
            "Shop - Packs": [
                (cls.BUY_PACK_TAROT, "Buy tarot pack"),
                (cls.BUY_PACK_PLANET, "Buy planet pack"),
                (cls.BUY_PACK_SPECTRAL, "Buy spectral pack"),
            ],
            "Shop - Other": [
                (cls.BUY_VOUCHER, "Buy voucher"),
            ],
            "Other": [
                (cls.NO_OP, "Do nothing"),
            ]
        }
