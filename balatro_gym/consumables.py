import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Any,List,Optional
from .cards import HandType
from .cards import Card, Rank, Suit, HandType


class ConsumableType(Enum):
    TAROT = "tarot"
    PLANET = "planet"
    SPECTRAL = "spectral"


@dataclass
class Consumable:
    """Single-use item the player can apply during a run."""
    name: str
    description: str
    consumable_type: ConsumableType
    effect: Callable[[Dict[str, Any]], Dict[str, Any]]
    data: Dict[str, Any] = field(default_factory=dict)

    def apply(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the consumable's effect to the game state."""
        print(f"[Consumable] Applying: {self.name}")
        return self.effect(game_state)


class ConsumableShop:
    """Simple container for consumables available in the shop."""

    def __init__(self):
        self.stock: list[Consumable] = []

    def add(self, consumable: Consumable):
        self.stock.append(consumable)

    def buy(self, index: int) -> Consumable:
        if index < 0 or index >= len(self.stock):
            raise IndexError("Invalid consumable index")
        return self.stock.pop(index)

    def __len__(self):
        return len(self.stock)

    def preview(self) -> list[str]:
        return [f"{i}: {c.name} ({c.consumable_type.value})" for i, c in enumerate(self.stock)]

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
            description="1 in 4 chance to add Red, Blue, Gold, or Purple seal to a random card in hand", # Clarified description
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
            description="Select 1 card in hand, convert another card in hand into a copy of the first", # Clarified description
            consumable_type=ConsumableType.TAROT,
            effect=TarotCards._death_effect
        )
        
        return tarots
    
    @staticmethod
    def _enhance_cards(state: Dict[str, Any], enhancement: str, count: int) -> Dict[str, Any]:
        """Generic enhancement effect for cards in hand."""
        hand = state.get("hand", []) # Assuming hand is List[Card]
        if not isinstance(hand, list) or not all(isinstance(c, Card) for c in hand):
            # print("Warning: 'hand' in state is not a list of Card objects for _enhance_cards.")
            return {"cards_enhanced": 0, "error": "Invalid hand data"}

        enhanced_count = 0
        # Iterate through a copy if modifying list structure, or ensure indices are valid
        # For simplicity, let's assume 'hand' contains Card objects with an 'enhancement' attribute
        
        # This part needs user interaction to select cards, which is complex for a gym.
        # For now, let's enhance the first 'count' unenhanced cards.
        cards_to_consider = [card for card in hand if not card.enhancement] # Filter for unenhanced cards
        
        for i in range(min(count, len(cards_to_consider))):
            # Find the original card in the actual hand to modify it
            # This assumes card objects are unique or have a way to be identified if copied
            # For a gym, it might be simpler to assume the 'hand' in state is the direct list of Card objects
            original_card_in_hand = cards_to_consider[i] # This is a reference if hand contains objects
            original_card_in_hand.enhancement = enhancement
            enhanced_count += 1
            
        return {"cards_enhanced": enhanced_count}
    
    @staticmethod
    def _fool_effect(state: Dict[str, Any]) -> Dict[str, Any]:
        """Copy last consumable used."""
        last_used = state.get("last_consumable_used") # Assuming this key holds a Consumable object
        if last_used and isinstance(last_used, Consumable) and \
           last_used.consumable_type in [ConsumableType.TAROT, ConsumableType.PLANET]:
            # Create a new Consumable object that is a copy
            # This might require a deepcopy or a more specific copy constructor if Consumables have complex state
            copied_consumable = Consumable(name=last_used.name,
                                           description=last_used.description,
                                           consumable_type=last_used.consumable_type,
                                           effect=last_used.effect, # Effect callable is copied
                                           data=dict(last_used.data)) # Copy data
            return {"create_consumables": [copied_consumable]} # Return as a list
        return {}
    
    @staticmethod
    def _high_priestess_effect(state: Dict[str, Any]) -> Dict[str, Any]:
        """Create up to 2 random Planet cards."""
        all_planets = list(PlanetCards.create_all().values())
        if not all_planets:
            return {"create_consumables": []}
        num_to_create = min(2, len(all_planets))
        created = random.sample(all_planets, num_to_create)
        return {"create_consumables": created}
    
    @staticmethod
    def _emperor_effect(state: Dict[str, Any]) -> Dict[str, Any]:
        """Create up to 2 random Tarot cards."""
        # Avoid recursive call if this tarot is chosen. Filter out "The Emperor" itself.
        all_tarots = [t for name, t in TarotCards.create_all().items() if name != "The Emperor"]
        if not all_tarots:
            return {"create_consumables": []}
        num_to_create = min(2, len(all_tarots))
        created = random.sample(all_tarots, num_to_create)
        return {"create_consumables": created}
    
    @staticmethod
    def _wheel_effect(state: Dict[str, Any]) -> Dict[str, Any]:
        """1 in 4 chance to add Red, Blue, Gold, or Purple seal to a random card in hand."""
        # This effect usually requires card selection by the player.
        # For the gym, let's apply to a random card if one exists.
        hand = state.get("hand", [])
        if not isinstance(hand, list) or not all(isinstance(c, Card) for c in hand):
            # print("Warning: 'hand' in state is not a list of Card objects for _wheel_effect.")
            return {}

        unsealed_cards = [card for card in hand if not card.seal] # Assuming Card has a 'seal' attribute
        if not unsealed_cards:
            return {}
            
        card_to_seal = random.choice(unsealed_cards)
        
        # Actual Balatro Wheel gives one of 4 specific items, not seals.
        # "1 in 4 chance for Joker, $15, Celestial Pack, Buffoon Pack"
        # If it's about seals, then the description "1 in 4 chance to add Red, Blue, Gold, or Purple seal" is what we implement.
        # Let's assume the description is the target.
        possible_seals = ["red", "blue", "gold", "purple"]
        if random.random() < 0.25: # 1 in 4 chance to actually get a seal
            chosen_seal = random.choice(possible_seals)
            card_to_seal.seal = chosen_seal
            return {"seal_added": chosen_seal, "card_sealed": str(card_to_seal)} # str(card) for logging
        return {"info": "Wheel of Fortune: no seal applied this time."}

    @staticmethod
    def _strength_effect(state: Dict[str, Any]) -> Dict[str, Any]:
        """Increases rank of up to 2 selected cards by 1."""
        hand = state.get("hand", [])
        if not isinstance(hand, list) or not all(isinstance(c, Card) for c in hand):
            # print("Warning: 'hand' in state is not a list of Card objects for _strength_effect.")
            return {"cards_upgraded": 0, "error": "Invalid hand data"}

        # Needs player selection. For gym, upgrade first 2 eligible (non-Ace) cards.
        upgraded_count = 0
        cards_considered_for_upgrade = 0
        
        for card_in_hand in hand:
            if cards_considered_for_upgrade >= 2:
                break
            
            if card_in_hand.rank.value < Rank.ACE.value: # Assuming Rank enum has .value and ACE is highest
                current_rank_value = card_in_hand.rank.value
                try:
                    next_rank_enum = Rank(current_rank_value + 1) # Assumes Rank values are contiguous integers
                    card_in_hand.rank = next_rank_enum
                    upgraded_count += 1
                except ValueError:
                    # This would happen if Rank values are not contiguous or current_rank_value + 1 is not a valid Rank value
                    # print(f"Warning: Could not find next rank for {card_in_hand.rank.name}")
                    # Fallback: Iterate through Rank enum to find next
                    found_next = False
                    for rank_member in Rank:
                        if rank_member.value == current_rank_value + 1:
                            card_in_hand.rank = rank_member
                            upgraded_count += 1
                            found_next = True
                            break
                    # if not found_next:
                        # print(f"Error: Still could not find next rank for {card_in_hand.rank.name} after iteration.")
                cards_considered_for_upgrade +=1
        
        return {"cards_upgraded": upgraded_count}
    
    @staticmethod
    def _death_effect(state: Dict[str, Any]) -> Dict[str, Any]:
        """Player selects one card, then selects another card to be converted into a copy of the first."""
        hand = state.get("hand", [])
        if not isinstance(hand, list) or not all(isinstance(c, Card) for c in hand):
            # print("Warning: 'hand' in state is not a list of Card objects for _death_effect.")
            return {"cards_converted": 0, "error": "Invalid hand data"}

        if len(hand) < 2:
            return {"cards_converted": 0, "info": "Not enough cards for Death effect."}

        # Needs player selection. For gym: copy first card to second.
        # This means hand[1] becomes a copy of hand[0].
        # The Card object would need a copy method or attributes to be copied manually.
        card_to_copy_from = hand[0]
        card_to_convert = hand[1]

        card_to_convert.rank = card_to_copy_from.rank
        card_to_convert.suit = card_to_copy_from.suit
        card_to_convert.enhancement = card_to_copy_from.enhancement # Copy enhancement
        card_to_convert.edition = card_to_copy_from.edition       # Copy edition
        card_to_convert.seal = card_to_copy_from.seal             # Copy seal
        # Any other relevant attributes of a Card should be copied too.
        
        return {"cards_converted": 1, "copied_card": str(card_to_copy_from), "converted_card": str(card_to_convert)}


# --- Planet Cards Implementation ---

# Base Level 1 stats for each hand type
# (Chips, Mult)
BASE_HAND_STATS: Dict[HandType, Dict[str, int]] = {
    HandType.HIGH_CARD: {"chips": 5, "mult": 1},
    HandType.PAIR: {"chips": 10, "mult": 2},
    HandType.TWO_PAIR: {"chips": 20, "mult": 2},
    HandType.THREE_OF_A_KIND: {"chips": 30, "mult": 3},
    HandType.STRAIGHT: {"chips": 30, "mult": 4},
    HandType.FLUSH: {"chips": 35, "mult": 4},
    HandType.FULL_HOUSE: {"chips": 40, "mult": 4},
    HandType.FOUR_OF_A_KIND: {"chips": 60, "mult": 5},
    HandType.STRAIGHT_FLUSH: {"chips": 100, "mult": 8},
    # Assuming ROYAL_FLUSH and FIVE_OF_A_KIND are distinct HandTypes
    # If ROYAL_FLUSH is just a specific Straight Flush, it might not have its own level.
    # Based on user's code, these are distinct.
    HandType.ROYAL_FLUSH: {"chips": 100, "mult": 8}, # Often same as Straight Flush L1
    HandType.FIVE_OF_A_KIND: {"chips": 70, "mult": 7}, # A strong, but non-standard hand
}

class PlanetCards:
    """All planet card effects (level up poker hands)."""
    
    # Data for planet effects: (HandTypeToLevel, PlanetName, {chips_bonus_per_level, mult_bonus_per_level})
    _PLANET_DATA = [
        (HandType.HIGH_CARD, "Pluto", {"chips_bonus": 15, "mult_bonus": 1}),
        (HandType.PAIR, "Mercury", {"chips_bonus": 15, "mult_bonus": 1}),
        (HandType.TWO_PAIR, "Uranus", {"chips_bonus": 20, "mult_bonus": 1}),
        (HandType.THREE_OF_A_KIND, "Venus", {"chips_bonus": 20, "mult_bonus": 2}),
        (HandType.STRAIGHT, "Saturn", {"chips_bonus": 30, "mult_bonus": 3}),
        (HandType.FLUSH, "Jupiter", {"chips_bonus": 15, "mult_bonus": 2}),
        (HandType.FULL_HOUSE, "Earth", {"chips_bonus": 25, "mult_bonus": 2}),
        (HandType.FOUR_OF_A_KIND, "Mars", {"chips_bonus": 30, "mult_bonus": 3}),
        (HandType.STRAIGHT_FLUSH, "Neptune", {"chips_bonus": 40, "mult_bonus": 4}),
        # User-defined mappings from their original snippet
        (HandType.ROYAL_FLUSH, "Planet X", {"chips_bonus": 50, "mult_bonus": 4}), 
        (HandType.FIVE_OF_A_KIND, "Ceres", {"chips_bonus": 35, "mult_bonus": 3}),
    ]

    @staticmethod
    def create_all() -> Dict[str, Consumable]:
        planets = {}
        
        for hand_type_to_level, planet_name, bonus_data in PlanetCards._PLANET_DATA:
            # Capture these specific values for the lambda
            ht_val = hand_type_to_level
            cb_val = bonus_data["chips_bonus"]
            mb_val = bonus_data["mult_bonus"]

            # Data for introspection (stored on the Consumable object)
            consumable_data_for_inspection = {
                "levels_hand_type": ht_val.name, # Store HandType name as string
                "chips_bonus_per_level": cb_val,
                "mult_bonus_per_level": mb_val
            }
            
            planets[planet_name] = Consumable(
                name=planet_name,
                description=f"Level up {ht_val.name.replace('_', ' ')}. (+{cb_val} Chips, +{mb_val} Mult to hand's base per level)",
                consumable_type=ConsumableType.PLANET,
                effect=lambda state, current_ht=ht_val, current_cb=cb_val, current_mb=mb_val: \
                       PlanetCards._level_up_logic(state, current_ht, current_cb, current_mb),
                data=consumable_data_for_inspection
            )
        return planets

    @staticmethod
    def _level_up_logic(state: Dict[str, Any], hand_to_level: HandType, 
                        chips_bonus: int, mult_bonus: int) -> Dict[str, Any]:
        """
        Core logic for leveling up a poker hand.
        Updates the hand's level, base chips, and base mult in the game state.
        """
        if 'hand_levels' not in state:
            state['hand_levels'] = {}

        # Initialize hand stats if not already present.
        # A hand starts at Level 1 with its BASE_HAND_STATS.
        # The first planet card usage makes it Level 2 and applies the first bonus.
        if hand_to_level not in state['hand_levels']:
            if hand_to_level not in BASE_HAND_STATS:
                # This should not happen if BASE_HAND_STATS and _PLANET_DATA are comprehensive
                # print(f"Error: Base stats for {hand_to_level.name} not found. Cannot level up.")
                return {"error": f"Base stats for {hand_to_level.name} not found."}
            
            base_stats = BASE_HAND_STATS[hand_to_level]
            state['hand_levels'][hand_to_level] = {
                'level': 1,  # Starts at Level 1
                'chips': base_stats['chips'],  # Current total chips for this hand type's base
                'mult': base_stats['mult']     # Current total mult for this hand type's base
            }
        
        # Apply the level up: increment level, add bonuses to current chips/mult
        current_hand_stats = state['hand_levels'][hand_to_level]
        
        current_hand_stats['level'] += 1
        current_hand_stats['chips'] += chips_bonus
        current_hand_stats['mult'] += mult_bonus
        
        return {
            "hand_leveled_up": hand_to_level.name,
            "new_level": current_hand_stats['level'],
            "current_hand_base_chips": current_hand_stats['chips'],
            "current_hand_base_mult": current_hand_stats['mult'],
            "info": f"{hand_to_level.name} leveled up to Lvl {current_hand_stats['level']} ({current_hand_stats['chips']} Chips, +{current_hand_stats['mult']} Mult)"
        }

# --- Consumable Shop Implementation ---
class ConsumableShop:
    """
    Manages the availability and offerings of consumables in the game.
    """
    def __init__(self):
        """
        Initializes the shop by loading all available Tarot and Planet cards.
        """
        self.all_tarots: Dict[str, Consumable] = TarotCards.create_all()
        self.all_planets: Dict[str, Consumable] = PlanetCards.create_all()
        # Potentially other consumable types like Spectral cards in the future
        # self.all_spectrals: Dict[str, Consumable] = SpectralCards.create_all() 
        
        self.current_shop_offerings: List[Consumable] = []
        self.shop_slots: int = 2 # Example: shop offers 2 items at a time (like Celestial/Standard packs)

    def refresh_shop_offerings(self, num_tarot_slots: int = 1, num_planet_slots: int = 1) -> None:
        """
        Refreshes the consumables available in the shop.
        This is a basic implementation; actual Balatro shop has pack mechanics.
        This could represent, for example, the contents of a "Celestial Pack" or general shop slots.

        Args:
            num_tarot_slots (int): Number of tarot card slots to offer.
            num_planet_slots (int): Number of planet card slots to offer.
        """
        self.current_shop_offerings = []
        
        # Offer Tarot cards
        available_tarots = list(self.all_tarots.values())
        if available_tarots:
            num_to_offer = min(num_tarot_slots, len(available_tarots))
            self.current_shop_offerings.extend(random.sample(available_tarots, num_to_offer))
            
        # Offer Planet cards
        available_planets = list(self.all_planets.values())
        if available_planets:
            num_to_offer = min(num_planet_slots, len(available_planets))
            self.current_shop_offerings.extend(random.sample(available_planets, num_to_offer))
            
        # Shuffle the final offerings if desired
        random.shuffle(self.current_shop_offerings)
        
        # print(f"Shop refreshed. Offerings: {[c.name for c in self.current_shop_offerings]}")

    def get_shop_offerings(self) -> List[Consumable]:
        """
        Returns the current list of consumables offered by the shop.
        """
        return self.current_shop_offerings

    def purchase_consumable(self, consumable_name: str, player_money: int, cost: int) -> Optional[Consumable]:
        """
        Allows a player to "purchase" a consumable from the current offerings.
        Removes it from shop offerings and deducts cost.

        Args:
            consumable_name (str): The name of the consumable to purchase.
            player_money (int): Current money of the player.
            cost (int): The cost of the consumable.

        Returns:
            Optional[Consumable]: The purchased Consumable object if successful, else None.
        """
        if player_money < cost:
            # print(f"Not enough money to buy {consumable_name}. Cost: {cost}, Have: {player_money}")
            return None

        for i, item in enumerate(self.current_shop_offerings):
            if item.name == consumable_name:
                purchased_item = self.current_shop_offerings.pop(i)
                # player_money -= cost # Money deduction would happen in the env
                # print(f"Purchased {purchased_item.name} for ${cost}.")
                return purchased_item
        
        # print(f"Consumable {consumable_name} not found in current shop offerings.")
        return None

# Example of how to get a Planet card and use its effect (for testing purposes)
if __name__ == '__main__':
    # This part is for demonstration and would not be in the final consumables.py
    # Mock Card, Rank, Suit, HandType for standalone testing
    class MockEnum(Enum):
        def __str__(self):
            return self.name
    
    class Rank(MockEnum): # Mocking Rank for test execution
        TWO = 2
        THREE = 3
        FOUR = 4
        FIVE = 5
        SIX = 6
        SEVEN = 7
        EIGHT = 8
        NINE = 9
        TEN = 10
        JACK = 11
        QUEEN = 12
        KING = 13
        ACE = 14
    
    class Suit(MockEnum): # Mocking Suit
        HEARTS = 1
        SPADES = 2
        DIAMONDS = 3
        CLUBS = 4


    class HandType(MockEnum): # Mocking HandType
        HIGH_CARD = "High Card"
        PAIR = "Pair"
        TWO_PAIR = "Two Pair"
        THREE_OF_A_KIND = "Three of a Kind"
        STRAIGHT = "Straight"
        FLUSH = "Flush"
        FULL_HOUSE = "Full House"
        FOUR_OF_A_KIND = "Four of a Kind"
        STRAIGHT_FLUSH = "Straight Flush"
        ROYAL_FLUSH = "Royal Flush"
        FIVE_OF_A_KIND = "Five of a Kind"

    @dataclass
    class Card: # Simplified Card for testing Tarot effects
        rank: Rank
        suit: Suit
        enhancement: Optional[str] = None
        edition: Optional[str] = None
        seal: Optional[str] = None

        def __str__(self):
            return f"{self.rank.name} of {self.suit.name}"


    # Re-define BASE_HAND_STATS here because HandType is mocked
    BASE_HAND_STATS: Dict[HandType, Dict[str, int]] = {
        HandType.HIGH_CARD: {"chips": 5, "mult": 1},
        HandType.PAIR: {"chips": 10, "mult": 2},
        HandType.TWO_PAIR: {"chips": 20, "mult": 2},
        HandType.THREE_OF_A_KIND: {"chips": 30, "mult": 3},
        HandType.STRAIGHT: {"chips": 30, "mult": 4},
        HandType.FLUSH: {"chips": 35, "mult": 4},
        HandType.FULL_HOUSE: {"chips": 40, "mult": 4},
        HandType.FOUR_OF_A_KIND: {"chips": 60, "mult": 5},
        HandType.STRAIGHT_FLUSH: {"chips": 100, "mult": 8},
        HandType.ROYAL_FLUSH: {"chips": 100, "mult": 8},
        HandType.FIVE_OF_A_KIND: {"chips": 70, "mult": 7},
    }


    # Test Planet Card leveling
    game_state: Dict[str, Any] = {"money": 50, "hand_levels": {}} # Increased money for shop test
    
    print("Initial game_state:", game_state)

    all_planet_cards = PlanetCards.create_all()
    mercury_card = all_planet_cards.get("Mercury")

    if mercury_card:
        print(f"\nUsing {mercury_card.name}: {mercury_card.description}")
        result = mercury_card.effect(game_state)
        print("Effect result:", result)
        print("Game state after 1st Mercury:", game_state['hand_levels'].get(HandType.PAIR))

        print(f"\nUsing {mercury_card.name} again...")
        result = mercury_card.effect(game_state)
        print("Effect result:", result)
        print("Game state after 2nd Mercury:", game_state['hand_levels'].get(HandType.PAIR))

    # Test a different planet card
    earth_card = all_planet_cards.get("Earth")
    if earth_card:
        print(f"\nUsing {earth_card.name}: {earth_card.description}")
        result = earth_card.effect(game_state)
        print("Effect result:", result)
        print("Game state after Earth:", game_state['hand_levels'].get(HandType.FULL_HOUSE))

    # Test Tarot Card - The Hermit
    tarot_cards = TarotCards.create_all()
    hermit_card = tarot_cards.get("The Hermit")
    if hermit_card:
        game_state["money"] = 15 # Reset money for this test
        print(f"\nUsing {hermit_card.name}: {hermit_card.description}")
        print("Money before Hermit:", game_state["money"])
        result = hermit_card.effect(game_state)
        game_state["money"] += result.get("money_add",0)
        print("Effect result:", result)
        print("Money after Hermit:", game_state["money"])
    
    # Test Tarot Card - Strength (needs mock cards in hand)
    strength_card = tarot_cards.get("Strength")
    if strength_card:
        print(f"\nUsing {strength_card.name}: {strength_card.description}")
        mock_hand_for_strength = [
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.KING, Suit.SPADES),
            Card(Rank.ACE, Suit.DIAMONDS) # Ace should not be upgraded
        ]
        game_state["hand"] = mock_hand_for_strength
        print(f"Hand before Strength: {[str(c) + ' (' + c.rank.name + ')' for c in game_state['hand']]}")
        result = strength_card.effect(game_state)
        print("Effect result:", result)
        print(f"Hand after Strength: {[str(c) + ' (' + c.rank.name + ')' for c in game_state['hand']]}")

    # Test ConsumableShop
    print("\n--- Testing ConsumableShop ---")
    shop = ConsumableShop()
    print(f"Shop initialized with {len(shop.all_tarots)} Tarots and {len(shop.all_planets)} Planets.")
    
    shop.refresh_shop_offerings(num_tarot_slots=1, num_planet_slots=1)
    offerings = shop.get_shop_offerings()
    print(f"Initial shop offerings: {[o.name for o in offerings]}")

    if offerings:
        item_to_buy_name = offerings[0].name
        item_cost = 3 # Example cost
        print(f"\nAttempting to buy '{item_to_buy_name}' for ${item_cost}. Player money: ${game_state['money']}")
        
        purchased = shop.purchase_consumable(item_to_buy_name, game_state['money'], item_cost)
        if purchased:
            game_state['money'] -= item_cost # Actual money deduction
            print(f"Successfully purchased: {purchased.name}. Remaining money: ${game_state['money']}")
            print(f"Player inventory (conceptually): {purchased.name}")
            # Example of using the purchased item if it's a planet
            if purchased.consumable_type == ConsumableType.PLANET:
                 print(f"Using purchased planet card: {purchased.name}")
                 effect_result = purchased.effect(game_state)
                 print(f"Effect of {purchased.name}: {effect_result.get('info')}")
                 print(f"Hand levels now: {game_state.get('hand_levels')}")

        else:
            print(f"Failed to purchase {item_to_buy_name}.")
        
        print(f"Shop offerings after attempt: {[o.name for o in shop.get_shop_offerings()]}")

    shop.refresh_shop_offerings(num_tarot_slots=2, num_planet_slots=0) # Offer only tarots
    print(f"Shop offerings (only tarots): {[o.name for o in shop.get_shop_offerings()]}")

