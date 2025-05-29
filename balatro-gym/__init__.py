# balatro_gym/__init__.py

__version__ = "0.1.0"

from gymnasium.envs.registration import register

from .env import BalatroEnv
from .state_formatter import StateFormatter
from .reward_shaper import RewardShaper

# Actions, cards, consumables, etc.
from .actions import Action
from .cards import Card, Deck, HandType, HandEvaluator, Rank, Suit
from .jokers import Joker, JokerRarity
from .consumables import Consumable, ConsumableType
from .vouchers import Voucher

# Register Gymnasium envs
register(
    id="Balatro-v0",
    entry_point="balatro_gym:BalatroEnv",
)
register(
    id="BalatroSmall-v0",
    entry_point="balatro_gym:BalatroSmallEnv",
)

__all__ = [
    "__version__",
    "BalatroEnv",
    "BalatroSmallEnv",
    "StateFormatter",
    "RewardShaper",
    "Action",
    "Card",
    "Deck",
    "HandType",
    "HandEvaluator",
    "Rank",
    "Suit",
    "Joker",
    "JokerRarity",
    "Consumable",
    "ConsumableType",
    "Voucher",
]
