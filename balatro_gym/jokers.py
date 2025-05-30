from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional
from enum import Enum
import random

class JokerRarity(Enum):
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    LEGENDARY = "legendary"

@dataclass
class Joker:
    name: str
    description: str
    rarity: JokerRarity
    cost: int
    effect: Callable
    data: Dict[str, Any] = None  # Store joker-specific data
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}

class JokerEffects:
    """Collection of all joker effects in the game."""
    
    @staticmethod
    def create_all_jokers() -> Dict[str, Joker]:
        """Create all jokers with their effects."""
        jokers = {}
        
        # Common Jokers
        jokers["Joker"] = Joker(
            name="Joker",
            description="+4 Mult",
            rarity=JokerRarity.COMMON,
            cost=2,
            effect=lambda state: {"mult_add": 4}
        )
        
        jokers["Greedy Joker"] = Joker(
            name="Greedy Joker",
            description="Every played Diamond gives +3 Mult",
            rarity=JokerRarity.COMMON,
            cost=4,
            effect=lambda state: {
                "mult_add": 3 * sum(1 for card in state.get("played_cards", []) 
                                  if card.suit.name == "DIAMONDS")
            }
        )
        
        jokers["Lusty Joker"] = Joker(
            name="Lusty Joker",
            description="Every played Heart gives +3 Mult",
            rarity=JokerRarity.COMMON,
            cost=4,
            effect=lambda state: {
                "mult_add": 3 * sum(1 for card in state.get("played_cards", []) 
                                  if card.suit.name == "HEARTS")
            }
        )
        
        jokers["Wrathful Joker"] = Joker(
            name="Wrathful Joker",
            description="Every played Spade gives +3 Mult",
            rarity=JokerRarity.COMMON,
            cost=4,
            effect=lambda state: {
                "mult_add": 3 * sum(1 for card in state.get("played_cards", []) 
                                  if card.suit.name == "SPADES")
            }
        )
        
        jokers["Gluttonous Joker"] = Joker(
            name="Gluttonous Joker",
            description="Every played Club gives +3 Mult",
            rarity=JokerRarity.COMMON,
            cost=4,
            effect=lambda state: {
                "mult_add": 3 * sum(1 for card in state.get("played_cards", []) 
                                  if card.suit.name == "CLUBS")
            }
        )
        
        # Uncommon Jokers
        jokers["Half Joker"] = Joker(
            name="Half Joker",
            description="+20 Mult if played hand has 3 or fewer cards",
            rarity=JokerRarity.UNCOMMON,
            cost=5,
            effect=lambda state: {
                "mult_add": 20 if len(state.get("played_cards", [])) <= 3 else 0
            }
        )
        
        jokers["Stencil Joker"] = Joker(
            name="Stencil Joker",
            description="X1 Mult for each empty Joker slot (up to 5)",
            rarity=JokerRarity.UNCOMMON,
            cost=8,
            effect=lambda state: {
                "mult_mult": 1 + (5 - len(state.get("jokers", [])))
            }
        )
        
        jokers["Raised Fist"] = Joker(
            name="Raised Fist",
            description="Adds double the rank of the lowest card held in hand to Mult",
            rarity=JokerRarity.UNCOMMON,
            cost=6,
            effect=lambda state: {
                "mult_add": 2 * min(card.rank.value for card in state.get("hand", [])) 
                           if state.get("hand") else 0
            }
        )
        
        jokers["Fibonacci"] = Joker(
            name="Fibonacci",
            description="+8 Mult if played hand has an Ace, 2, 3, 5, or 8",
            rarity=JokerRarity.UNCOMMON,
            cost=6,
            effect=lambda state: {
                "mult_add": 8 if any(card.rank.value in [14, 2, 3, 5, 8] 
                                   for card in state.get("played_cards", [])) else 0
            }
        )
        
        # Rare Jokers
        jokers["Baseball Card"] = Joker(
            name="Baseball Card",
            description="Uncommon Jokers give X1.5 Mult",
            rarity=JokerRarity.RARE,
            cost=8,
            effect=lambda state: {
                "mult_mult": 1.5 ** sum(1 for j in state.get("jokers", []) 
                                      if j.rarity == JokerRarity.UNCOMMON)
            }
        )
        
        jokers["Ancient Joker"] = Joker(
            name="Ancient Joker",
            description="Each played card with a suit that matches the previous gives X1.5 Mult",
            rarity=JokerRarity.RARE,
            cost=8,
            effect=JokerEffects._ancient_joker_effect
        )
        
        jokers["Vampire"] = Joker(
            name="Vampire",
            description="Removes card enhancements and gains X0.2 Mult per enhancement removed",
            rarity=JokerRarity.RARE,
            cost=10,
            effect=JokerEffects._vampire_effect
        )
        
        # Legendary Jokers
        jokers["Triboulet"] = Joker(
            name="Triboulet",
            description="Played Kings and Queens give X2 Mult",
            rarity=JokerRarity.LEGENDARY,
            cost=15,
            effect=lambda state: {
                "mult_mult": 2 ** sum(1 for card in state.get("played_cards", []) 
                                     if card.rank.value in [12, 13])
            }
        )
        
        jokers["Baron"] = Joker(
            name="Baron",
            description="Kings in hand give X1.5 Mult",
            rarity=JokerRarity.LEGENDARY,
            cost=15,
            effect=lambda state: {
                "mult_mult": 1.5 ** sum(1 for card in state.get("hand", []) 
                                       if card.rank.value == 13)
            }
        )
        
        jokers["Mime"] = Joker(
            name="Mime",
            description="Retrigger all card abilities",
            rarity=JokerRarity.LEGENDARY,
            cost=12,
            effect=lambda state: {"retrigger_cards": True}
        )
        
        return jokers
    
    @staticmethod
    def _ancient_joker_effect(state: Dict) -> Dict:
        """Calculate Ancient Joker effect."""
        mult_mult = 1.0
        played_cards = state.get("played_cards", [])
        
        for i in range(1, len(played_cards)):
            if played_cards[i].suit == played_cards[i-1].suit:
                mult_mult *= 1.5
        
        return {"mult_mult": mult_mult}
    
    @staticmethod
    def _vampire_effect(state: Dict) -> Dict:
        """Calculate Vampire effect and modify cards."""
        played_cards = state.get("played_cards", [])
        enhancements_removed = 0
        
        for card in played_cards:
            if card.enhancement:
                card.enhancement = None
                enhancements_removed += 1
        
        # Store cumulative multiplier in joker data
        joker = state.get("current_joker")
        if joker:
            current_mult = joker.data.get("vampire_mult", 1.0)
            joker.data["vampire_mult"] = current_mult + (0.2 * enhancements_removed)
            return {"mult_mult": joker.data["vampire_mult"]}
        
        return {"mult_mult": 1.0}

class JokerShop:
    """Manages joker generation for the shop."""
    
    def __init__(self):
        self.all_jokers = JokerEffects.create_all_jokers()
        self.rarity_weights = {
            JokerRarity.COMMON: 70,
            JokerRarity.UNCOMMON: 25,
            JokerRarity.RARE: 4,
            JokerRarity.LEGENDARY: 1
        }
    
    def generate_shop_jokers(self, count: int = 2, ante: int = 1) -> List[Joker]:
        """Generate jokers for the shop based on ante."""
        shop_jokers = []
        
        # Adjust weights based on ante
        adjusted_weights = self.rarity_weights.copy()
        if ante >= 3:
            adjusted_weights[JokerRarity.UNCOMMON] += 5
            adjusted_weights[JokerRarity.RARE] += 2
        if ante >= 5:
            adjusted_weights[JokerRarity.RARE] += 3
            adjusted_weights[JokerRarity.LEGENDARY] += 1
        
        # Generate jokers
        for _ in range(count):
            rarity = self._weighted_random_rarity(adjusted_weights)
            available = [j for j in self.all_jokers.values() 
                        if j.rarity == rarity and j not in shop_jokers]
            
            if available:
                joker = random.choice(available)
                shop_jokers.append(joker)
        
        return shop_jokers
    
    def _weighted_random_rarity(self, weights: Dict[JokerRarity, int]) -> JokerRarity:
        """Select rarity based on weights."""
        total = sum(weights.values())
        r = random.uniform(0, total)
        
        cumulative = 0
        for rarity, weight in weights.items():
            cumulative += weight
            if r <= cumulative:
                return rarity
        
        return JokerRarity.COMMON
