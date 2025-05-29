import numpy as np
from typing import List, Dict, Any
from .actions import Action

class StateFormatter:
    """Formats game state into text descriptions for LLM agents."""
    
    def format_state(self, env) -> str:
        """Create a comprehensive text description of the game state."""
        sections = []
        
        # Game progress
        sections.append(f"=== GAME PROGRESS ===")
        sections.append(f"Ante: {env.current_ante}")
        sections.append(f"Blind: {env.current_blind.title()} (Target: {env.target_score} | Current: {env.current_score})")
        sections.append(f"Money: ${env.money}")
        sections.append("")
        
        # Current situation
        if env.in_shop:
            sections.append("=== SHOP ===")
            sections.append(self._format_shop(env))
        else:
            sections.append("=== CURRENT HAND ===")
            sections.append(f"Cards: {self._format_hand(env.hand)}")
            sections.append(f"Hands remaining: {env.hands_remaining}")
            sections.append(f"Discards remaining: {env.discards_remaining}")
        
        sections.append("")
        
        # Jokers
        if env.jokers:
            sections.append("=== JOKERS ===")
            for i, joker in enumerate(env.jokers):
                effect_desc = joker.description
                if hasattr(joker, 'data') and joker.data:
                    # Add dynamic data like Vampire's current multiplier
                    if 'vampire_mult' in joker.data:
                        effect_desc += f" (Current: X{joker.data['vampire_mult']:.1f})"
                    if joker.data.get('negative'):
                        effect_desc += " [NEGATIVE]"
                    if joker.data.get('polychrome'):
                        effect_desc += " [POLYCHROME]"
                sections.append(f"{i+1}. {joker.name} - {effect_desc}")
        else:
            sections.append("=== JOKERS ===")
            sections.append("No jokers owned")
        
        sections.append("")
        
        # Consumables
        sections.append("=== CONSUMABLES ===")
        if env.consumables:
            for i, consumable in enumerate(env.consumables):
                sections.append(f"{i+1}. {consumable.name} ({consumable.consumable_type.value}) - {consumable.description}")
        else:
            sections.append("Empty")
        
        sections.append("")
        
        # Hand levels (if any upgraded)
        upgraded_hands = [(ht, level) for ht, level in env.hand_levels.items() if level > 1]
        if upgraded_hands:
            sections.append("=== HAND LEVELS ===")
            for hand_type, level in upgraded_hands:
                sections.append(f"{hand_type.name.replace('_', ' ')}: Level {level}")
            sections.append("")
        
        sections.append("")
        
        # Recent history
        if env.action_history:
            sections.append("=== RECENT ACTIONS ===")
            for action in list(env.action_history)[-5:]:
                sections.append(f"- {action['action']}")
        
        return "\n".join(sections)
    
    def format_legal_actions(self, mask: np.ndarray) -> str:
        """Format legal actions in a structured way for LLM understanding."""
        action_groups = Action.get_action_groups()
        lines = ["=== LEGAL ACTIONS ==="]
        
        for group_name, actions in action_groups.items():
            group_actions = []
            for action_idx, description in actions:
                if mask[action_idx]:
                    group_actions.append(f"  [{action_idx}] {description}")
            
            if group_actions:
                lines.append(f"\n{group_name}:")
                lines.extend(group_actions)
        
        return "\n".join(lines)
    
    def _format_hand(self, hand: List) -> str:
        """Format a hand of cards."""
        if not hand:
            return "Empty"
        
        # Group by suit for readability
        by_suit = {}
        for card in hand:
            suit_name = str(card.suit)
            if suit_name not in by_suit:
                by_suit[suit_name] = []
            by_suit[suit_name].append(str(card))
        
        # Format as "♠: KS, 10S | ♥: AH, 5H"
        parts = []
        for suit, cards in by_suit.items():
            parts.append(f"{suit}: {', '.join(cards)}")
        
        return " | ".join(parts)
    
    def _format_shop(self, env) -> str:
        """Format shop contents."""
        lines = []
        
        # Jokers
        if env.shop_jokers:
            lines.append("Jokers:")
            for i, joker in enumerate(env.shop_jokers):
                if joker:
                    discount = env.voucher_effects.get("shop_discount", 0)
                    price = int(joker.cost * (1 - discount))
                    lines.append(f"  Slot {i}: {joker.name} - ${price} - {joker.description}")
                else:
                    lines.append(f"  Slot {i}: [SOLD]")
        
        # Packs
        lines.append("\nPacks:")
        pack_price = int(4 * (1 - env.voucher_effects.get("shop_discount", 0)))
        for pack_type, available in env.shop_packs.items():
            if available:
                lines.append(f"  {pack_type.title()} Pack - ${pack_price}")
        
        # Voucher
        if env.shop_voucher:
            lines.append(f"\nVoucher:")
            lines.append(f"  {env.shop_voucher.name} - ${env.shop_voucher.cost}")
            lines.append(f"    {env.shop_voucher.description}")
        
        # Playing cards (if available)
        if env.shop_cards:
            lines.append("\nPlaying Cards:")
            card_price = int(3 * (1 - env.voucher_effects.get("shop_discount", 0)))
            for i, card in enumerate(env.shop_cards):
                lines.append(f"  Card {i}: {card} - ${card_price}")
        
        lines.append(f"\nYour Money: ${env.money}")
        lines.append("(Use NO_OP action to leave shop)")
        
        return "\n".join(lines)
