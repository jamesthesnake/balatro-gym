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
                sections.append(f"{i+1}. {joker.get('name', 'Unknown')} - {joker.get('effect', 'No description')}")
        else:
            sections.append("=== JOKERS ===")
            sections.append("No jokers owned")
        
        sections.append("")
        
        # Consumable
        sections.append("=== CONSUMABLE ===")
        if env.consumable:
            sections.append(f"{env.consumable.get('name', 'Unknown')} - {env.consumable.get('effect', 'No description')}")
        else:
            sections.append("Empty")
        
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
    
    def _format_hand(self, hand: List[Dict]) -> str:
        """Format a hand of cards."""
        if not hand:
            return "Empty"
        
        # TODO: Implement proper card formatting
        # For now, just return placeholder
        return f"{len(hand)} cards"
    
    def _format_shop(self, env) -> str:
        """Format shop contents."""
        lines = []
        
        # Jokers
        if env.shop_jokers:
            lines.append("Jokers:")
            for i, joker in enumerate(env.shop_jokers):
                if joker:
                    lines.append(f"  Slot {i}: {joker['name']} - ${joker['cost']}")
                else:
                    lines.append(f"  Slot {i}: [SOLD]")
        
        # Packs
        lines.append("\nPacks:")
        for pack_type, available in env.shop_packs.items():
            if available:
                lines.append(f"  {pack_type.title()} Pack - $4")
        
        # Voucher
        if env.shop_voucher:
            lines.append(f"\nVoucher:")
            lines.append(f"  {env.shop_voucher['name']} - ${env.shop_voucher['cost']}")
        
        return "\n".join(lines)
