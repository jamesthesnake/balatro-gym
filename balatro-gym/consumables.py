from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional
from enum import Enum
import random
from .cards import Card, Rank, Suit, HandType

class ConsumableType(Enum):
    TAROT = "tarot"
    PLANET = "planet"
    SPECTRAL = "spectral"

@dataclass
class Consumable:
    name: str
    description: str
    consumable_type: ConsumableType
    effect: Callable
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}

class TarotCards:
    """All tarot card effects."""
    
    @staticmethod
    def create_all() -> Dict[str, Consumable]:
        tarots = {}
        
        tarots["The Fool"] = Consumable(
            name="The Fool",
            description="Creates a copy of the last tarot or planet card used",
            consumable_type=ConsumableType.TAROT,
            effect=TarotCards._fool_effect
        )
        
        tarots["The Magician"] = Consumable(
            name="The Magician",
            description="Enhances up to 2 cards into Lucky cards",
            consumable_type=ConsumableType.TAROT,
            effect=lambda state: TarotCards._enhance_cards(state, "lucky", 2)
        )
        
        tarots["The High Priestess"] = Consumable(
            name="The High Priestess",
            description="Creates up to 2 random Planet cards",
            consumable_type=ConsumableType.TAROT,
            effect=TarotCards._high_priestess_effect
        )
        
        tarots["The Empress"] = Consumable(
            name="The Empress",
            description="Enhances up to 2 cards into Mult cards",
            consumable_type=ConsumableType.TAROT,
            effect=lambda state: TarotCards._enhance_cards(state, "mult", 2)
        )
        
        tarots["The Emperor"] = Consumable(
            name="The Emperor",
            description="Creates up to 2 random Tarot cards",
            consumable_type=ConsumableType.TAROT,
            effect=TarotCards._emperor_effect
        )
        
        tarots["The Hierophant"] = Consumable(
            name="The Hierophant",
            description="Enhances up to 2 cards into Bonus cards",
            consumable_type=ConsumableType.TAROT,
            effect=lambda state: TarotCards._enhance_cards(state, "bonus", 2)
        )
        
        tarots["The Lovers"] = Consumable(
            name="The Lovers",
            description="Enhances up to 1 card into a Wild card",
            consumable_type=ConsumableType.TAROT,
            effect=lambda state: TarotCards._enhance_cards(state, "wild", 1)
        )
        
        tarots["The Chariot"] = Consumable(
            name="The Chariot",
            description="Enhances up to 1 card into a Steel card",
            consumable_type=ConsumableType.TAROT,
            effect=lambda state: TarotCards._enhance_cards(state, "steel", 1)
        )
        
        tarots["Justice"] = Consumable(
            name="Justice",
            description="Enhances up to 1 card into a Glass card",
            consumable_type=ConsumableType.TAROT,
            effect=lambda state: TarotCards._enhance_cards(state, "glass", 1)
        )
        
        tarots["The Hermit"] = Consumable(
            name="The Hermit",
            description="Doubles money (up to $20)",
            consumable_type=ConsumableType.TAROT,
            effect=lambda state: {"money_add": min(state.get("money", 0), 20)}
        )
        
        tarots["The Wheel"] = Consumable(
            name="The Wheel",
            description="1 in 4 chance to add Red, Blue, Gold, or Purple seal",
            consumable_type=ConsumableType.TAROT,
            effect=TarotCards._wheel_effect
        )
        
        tarots["Strength"] = Consumable(
            name="Strength",
            description="Increases rank of up to 2 cards by 1",
            consumable_type=ConsumableType.TAROT,
            effect=TarotCards._strength_effect
        )
        
        tarots["Death"] = Consumable(
            name="Death",
            description="Select 2 cards to convert into copies of each other",
            consumable_type=ConsumableType.TAROT,
            effect=TarotCards._death_effect
        )
        
        return tarots
    
    @staticmethod
    def _enhance_cards(state: Dict, enhancement: str, count: int) -> Dict:
        """Generic enhancement effect."""
        hand = state.get("hand", [])
        enhanced = 0
        
        for card in hand[:count]:
            if not card.enhancement:
                card.enhancement = enhancement
                enhanced += 1
        
        return {"cards_enhanced": enhanced}
    
    @staticmethod
    def _fool_effect(state: Dict) -> Dict:
        """Copy last consumable used."""
        last_used = state.get("last_consumable")
        if last_used and last_used.consumable_type in [ConsumableType.TAROT, ConsumableType.PLANET]:
            return {"create_consumable": last_used}
        return {}
    
    @staticmethod
    def _high_priestess_effect(state: Dict) -> Dict:
        """Create planet cards."""
        planets = list(PlanetCards.create_all().values())
        created = random.sample(planets, min(2, len(planets)))
        return {"create_consumables": created}
    
    @staticmethod
    def _emperor_effect(state: Dict) -> Dict:
        """Create tarot cards."""
        tarots = list(TarotCards.create_all().values())
        created = random.sample(tarots, min(2, len(tarots)))
        return {"create_consumables": created}
    
    @staticmethod
    def _wheel_effect(state: Dict) -> Dict:
        """Add random seal to a card."""
        hand = state.get("hand", [])
        if hand:
            card = random.choice(hand)
            card.seal = random.choice(["red", "blue", "gold", "purple"])
            return {"seal_added": card.seal}
        return {}
    
    @staticmethod
    def _strength_effect(state: Dict) -> Dict:
        """Increase card ranks."""
        hand = state.get("hand", [])
        upgraded = 0
        
        for card in hand[:2]:
            if card.rank.value < 14:  # Not an Ace
                # Find next rank
                for rank in Rank:
                    if rank.value == card.rank.value + 1:
                        card.rank = rank
                        upgraded += 1
                        break
        
        return {"cards_upgraded": upgraded}
    
    @staticmethod
    def _death_effect(state: Dict) -> Dict:
        """Convert cards to copies."""
        hand = state.get("hand", [])
        if len(hand) >= 2:
            # For simplicity, copy first card to second
            hand[1].rank = hand[0].rank
            hand[1].suit = hand[0].suit
            return {"cards_converted": 2}
        return {}

class PlanetCards:
    """All planet card effects (level up poker hands)."""
    
    @staticmethod
    def create_all() -> Dict[str, Consumable]:
        planets = {}
        hand_types = [
            (HandType.HIGH_CARD, "Pluto"),
            (HandType.PAIR, "Mercury"),
            (HandType.TWO_PAIR, "Uranus"),
            (HandType.THREE_OF_A_KIND, "Venus"),
            (HandType.STRAIGHT, "Saturn"),
            (HandType.FLUSH, "Jupiter"),
            (HandType.FULL_HOUSE, "Earth"),
            (HandType.FOUR_OF_A_KIND, "Mars"),
            (HandType.STRAIGHT_FLUSH, "Neptune"),
            (HandType.ROYAL_FLUSH, "Planet X"),
            (HandType.FIVE_OF_A_KIND, "Ceres"),
        ]
        
        for hand_type, planet_name in hand_types:
            planets[planet_name] = Consumable(
                name=planet_name,
                description=f"Level up {hand_type.name.replace('_',
