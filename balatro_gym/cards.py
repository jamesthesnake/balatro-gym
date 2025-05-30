from enum import IntEnum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random

class Suit(IntEnum):
    SPADES = 0
    HEARTS = 1
    CLUBS = 2
    DIAMONDS = 3
    
    def __str__(self):
        return ["♠", "♥", "♣", "♦"][self.value]

class Rank(IntEnum):
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
    
    def __str__(self):
        if self.value <= 10:
            return str(self.value)
        return ["J", "Q", "K", "A"][self.value - 11]

@dataclass
class Card:
    rank: Rank
    suit: Suit
    enhancement: Optional[str] = None  # glass, steel, gold, etc.
    seal: Optional[str] = None  # red, blue, gold, purple
    
    def __str__(self):
        base = f"{self.rank}{self.suit}"
        if self.enhancement:
            base = f"[{self.enhancement[0].upper()}]{base}"
        if self.seal:
            base = f"{base}*"
        return base
    
    def get_chip_value(self) -> int:
        """Base chip value of the card."""
        if self.rank <= Rank.TEN:
            return self.rank.value
        return 10  # Face cards worth 10

class Deck:
    def __init__(self):
        self.cards = []
        self.reset()
    
    def reset(self):
        """Create standard 52-card deck."""
        self.cards = []
        for suit in Suit:
            for rank in Rank:
                self.cards.append(Card(rank, suit))
        self.shuffle()
    
    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)
    
    def draw(self, count: int = 1) -> List[Card]:
        """Draw cards from the deck."""
        drawn = []
        for _ in range(min(count, len(self.cards))):
            drawn.append(self.cards.pop())
        return drawn
    
    def add_cards(self, cards: List[Card]):
        """Add cards back to deck (for discards)."""
        self.cards.extend(cards)
        self.shuffle()

class HandType(IntEnum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10
    FIVE_OF_A_KIND = 11  # With joker effects
    
    def get_base_chips(self) -> int:
        """Base chip value for each hand type."""
        base_values = {
            HandType.HIGH_CARD: 5,
            HandType.PAIR: 10,
            HandType.TWO_PAIR: 20,
            HandType.THREE_OF_A_KIND: 30,
            HandType.STRAIGHT: 40,
            HandType.FLUSH: 40,
            HandType.FULL_HOUSE: 50,
            HandType.FOUR_OF_A_KIND: 80,
            HandType.STRAIGHT_FLUSH: 120,
            HandType.ROYAL_FLUSH: 150,
            HandType.FIVE_OF_A_KIND: 200,
        }
        return base_values[self]
    
    def get_base_mult(self) -> int:
        """Base multiplier for each hand type."""
        base_mults = {
            HandType.HIGH_CARD: 1,
            HandType.PAIR: 2,
            HandType.TWO_PAIR: 2,
            HandType.THREE_OF_A_KIND: 3,
            HandType.STRAIGHT: 4,
            HandType.FLUSH: 4,
            HandType.FULL_HOUSE: 4,
            HandType.FOUR_OF_A_KIND: 6,
            HandType.STRAIGHT_FLUSH: 8,
            HandType.ROYAL_FLUSH: 10,
            HandType.FIVE_OF_A_KIND: 12,
        }
        return base_mults[self]

class HandEvaluator:
    @staticmethod
    def evaluate(cards: List[Card]) -> Tuple[HandType, List[Card]]:
        """Evaluate poker hand and return type with scoring cards."""
        if len(cards) < 5:
            return HandType.HIGH_CARD, cards
        
        # Sort by rank
        sorted_cards = sorted(cards, key=lambda c: c.rank, reverse=True)
        
        # Count ranks and suits
        rank_counts = {}
        suit_counts = {}
        for card in sorted_cards:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
            suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
        
        # Check for various hands
        is_flush = any(count >= 5 for count in suit_counts.values())
        is_straight, straight_cards = HandEvaluator._check_straight(sorted_cards)
        
        # Get rank groups
        rank_groups = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
        
        # Five of a kind (with wild cards/jokers)
        if rank_groups[0][1] >= 5:
            scoring_cards = [c for c in sorted_cards if c.rank == rank_groups[0][0]][:5]
            return HandType.FIVE_OF_A_KIND, scoring_cards
        
        # Straight flush / Royal flush
        if is_straight and is_flush:
            flush_suit = max(suit_counts.items(), key=lambda x: x[1])[0]
            flush_cards = [c for c in sorted_cards if c.suit == flush_suit]
            is_straight_flush, sf_cards = HandEvaluator._check_straight(flush_cards)
            if is_straight_flush:
                if sf_cards[0].rank == Rank.ACE:
                    return HandType.ROYAL_FLUSH, sf_cards
                return HandType.STRAIGHT_FLUSH, sf_cards
        
        # Four of a kind
        if rank_groups[0][1] == 4:
            scoring_cards = [c for c in sorted_cards if c.rank == rank_groups[0][0]]
            scoring_cards.append(next(c for c in sorted_cards if c.rank != rank_groups[0][0]))
            return HandType.FOUR_OF_A_KIND, scoring_cards
        
        # Full house
        if rank_groups[0][1] == 3 and rank_groups[1][1] >= 2:
            scoring_cards = [c for c in sorted_cards if c.rank == rank_groups[0][0]]
            scoring_cards.extend([c for c in sorted_cards if c.rank == rank_groups[1][0]][:2])
            return HandType.FULL_HOUSE, scoring_cards
        
        # Flush
        if is_flush:
            flush_suit = max(suit_counts.items(), key=lambda x: x[1])[0]
            scoring_cards = [c for c in sorted_cards if c.suit == flush_suit][:5]
            return HandType.FLUSH, scoring_cards
        
        # Straight
        if is_straight:
            return HandType.STRAIGHT, straight_cards
        
        # Three of a kind
        if rank_groups[0][1] == 3:
            scoring_cards = [c for c in sorted_cards if c.rank == rank_groups[0][0]]
            kickers = [c for c in sorted_cards if c.rank != rank_groups[0][0]][:2]
            scoring_cards.extend(kickers)
            return HandType.THREE_OF_A_KIND, scoring_cards
        
        # Two pair
        if rank_groups[0][1] == 2 and rank_groups[1][1] == 2:
            scoring_cards = [c for c in sorted_cards if c.rank == rank_groups[0][0]]
            scoring_cards.extend([c for c in sorted_cards if c.rank == rank_groups[1][0]])
            kicker = next(c for c in sorted_cards if c.rank not in [rank_groups[0][0], rank_groups[1][0]])
            scoring_cards.append(kicker)
            return HandType.TWO_PAIR, scoring_cards
        
        # Pair
        if rank_groups[0][1] == 2:
            scoring_cards = [c for c in sorted_cards if c.rank == rank_groups[0][0]]
            kickers = [c for c in sorted_cards if c.rank != rank_groups[0][0]][:3]
            scoring_cards.extend(kickers)
            return HandType.PAIR, scoring_cards
        
        # High card
        return HandType.HIGH_CARD, sorted_cards[:5]
    
    @staticmethod
    def _check_straight(cards: List[Card]) -> Tuple[bool, List[Card]]:
        """Check if cards contain a straight."""
        if len(cards) < 5:
            return False, []
        
        # Check for regular straight
        for i in range(len(cards) - 4):
            straight = True
            straight_cards = [cards[i]]
            
            for j in range(1, 5):
                if i + j >= len(cards) or cards[i].rank - cards[i + j].rank != j:
                    straight = False
                    break
                straight_cards.append(cards[i + j])
            
            if straight:
                return True, straight_cards
        
        # Check for A-2-3-4-5 straight
        if cards[0].rank == Rank.ACE:
            low_straight = [cards[0]]  # Ace
            needed = [Rank.FIVE, Rank.FOUR, Rank.THREE, Rank.TWO]
            for card in cards[1:]:
                if needed and card.rank == needed[0]:
                    low_straight.append(card)
                    needed.pop(0)
            
            if len(low_straight) == 5:
                return True, low_straight
        
        return False, []
