from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional
import random

@dataclass
class Voucher:
    name: str
    description: str
    cost: int
    effect: Callable
    tier: int = 1  # Some vouchers have upgraded versions
    upgrade: Optional[str] = None  # Name of upgraded version
    
class VoucherEffects:
    """All voucher effects in the game."""
    
    @staticmethod
    def create_all() -> Dict[str, Voucher]:
        vouchers = {}
        
        # Tier 1 Vouchers
        vouchers["Overstock"] = Voucher(
            name="Overstock",
            description="+1 card slot in shop",
            cost=10,
            effect=lambda state: {"shop_slots_add": 1},
            upgrade="Overstock Plus"
        )
        
        vouchers["Clearance Sale"] = Voucher(
            name="Clearance Sale",
            description="All cards and packs in shop are 25% off",
            cost=10,
            effect=lambda state: {"shop_discount": 0.25},
            upgrade="Liquidation"
        )
        
        vouchers["Hone"] = Voucher(
            name="Hone",
            description="+$2 per discard",
            cost=10,
            effect=lambda state: {"discard_money": 2},
            upgrade="Glow Up"
        )
        
        vouchers["Reroll Surplus"] = Voucher(
            name="Reroll Surplus",
            description="Rerolls cost $1 less",
            cost=10,
            effect=lambda state: {"reroll_cost_reduction": 1},
            upgrade="Reroll Glut"
        )
        
        vouchers["Crystal Ball"] = Voucher(
            name="Crystal Ball",
            description="+1 consumable slot",
            cost=10,
            effect=lambda state: {"consumable_slots_add": 1},
            upgrade="Omen Globe"
        )
        
        vouchers["Telescope"] = Voucher(
            name="Telescope",
            description="Celestrium booster packs have 1 extra card",
            cost=10,
            effect=lambda state: {"planet_pack_extra": 1},
            upgrade="Observatory"
        )
        
        vouchers["Grabber"] = Voucher(
            name="Grabber",
            description="+1 hand per round",
            cost=10,
            effect=lambda state: {"hands_add": 1},
            upgrade="Nacho Tong"
        )
        
        vouchers["Wasteful"] = Voucher(
            name="Wasteful",
            description="+1 discard per round",
            cost=10,
            effect=lambda state: {"discards_add": 1},
            upgrade="Recyclomancy"
        )
        
        vouchers["Seed Money"] = Voucher(
            name="Seed Money",
            description="Raise base earn cap to $25",
            cost=10,
            effect=lambda state: {"interest_cap": 25},
            upgrade="Money Tree"
        )
        
        vouchers["Blank"] = Voucher(
            name="Blank",
            description="Does nothing?",
            cost=10,
            effect=lambda state: {},  # Actually fills Joker slots with random commons
            upgrade="Antimatter"
        )
        
        vouchers["Magic Trick"] = Voucher(
            name="Magic Trick",
            description="Playing cards can be purchased from shop",
            cost=10,
            effect=lambda state: {"cards_in_shop": True},
            upgrade="Illusion"
        )
        
        vouchers["Hieroglyph"] = Voucher(
            name="Hieroglyph",
            description="-1 Ante, -1 hand per round",
            cost=10,
            effect=lambda state: {"ante_reduction": 1, "hands_add": -1},
            upgrade="Petroglyph"
        )
        
        vouchers["Director's Cut"] = Voucher(
            name="Director's Cut",
            description="Reroll Boss Blind 1 time per ante, $0 cost",
            cost=10,
            effect=lambda state: {"boss_reroll": 1},
            upgrade="Retcon"
        )
        
        vouchers["Paint Brush"] = Voucher(
            name="Paint Brush",
            description="+1 hand size",
            cost=10,
            effect=lambda state: {"hand_size_add": 1},
            upgrade="Palette"
        )
        
        # Tier 2 Vouchers (Upgrades)
        vouchers["Overstock Plus"] = Voucher(
            name="Overstock Plus",
            description="+1 card slot in shop",
            cost=10,
            effect=lambda state: {"shop_slots_add": 1},
            tier=2
        )
        
        vouchers["Liquidation"] = Voucher(
            name="Liquidation",
            description="All cards and packs in shop are 50% off",
            cost=10,
            effect=lambda state: {"shop_discount": 0.5},
            tier=2
        )
        
        vouchers["Glow Up"] = Voucher(
            name="Glow Up",
            description="+$4 per discard",
            cost=10,
            effect=lambda state: {"discard_money": 4},
            tier=2
        )
        
        vouchers["Reroll Glut"] = Voucher(
            name="Reroll Glut",
            description="Rerolls cost $2 less, gain free reroll per shop",
            cost=10,
            effect=lambda state: {"reroll_cost_reduction": 2, "free_rerolls": 1},
            tier=2
        )
        
        vouchers["Omen Globe"] = Voucher(
            name="Omen Globe",
            description="Spectral packs may appear in shop",
            cost=10,
            effect=lambda state: {"spectral_packs_enabled": True},
            tier=2
        )
        
        vouchers["Observatory"] = Voucher(
            name="Observatory",
            description="Planet cards in your consumable area give X1.5 Mult",
            cost=10,
            effect=lambda state: {"planet_card_mult": 1.5},
            tier=2
        )
        
        vouchers["Nacho Tong"] = Voucher(
            name="Nacho Tong",
            description="+1 hand per round",
            cost=10,
            effect=lambda state: {"hands_add": 1},
            tier=2
        )
        
        vouchers["Recyclomancy"] = Voucher(
            name="Recyclomancy",
            description="+1 discard per round, all discards give X2 Mult",
            cost=10,
            effect=lambda state: {"discards_add": 1, "discard_mult": 2},
            tier=2
        )
        
        vouchers["Money Tree"] = Voucher(
            name="Money Tree",
            description="Raise base earn cap to $50",
            cost=10,
            effect=lambda state: {"interest_cap": 50},
            tier=2
        )
        
        vouchers["Antimatter"] = Voucher(
            name="Antimatter",
            description="+1 Joker slot",
            cost=10,
            effect=lambda state: {"joker_slots_add": 1},
            tier=2
        )
        
        vouchers["Illusion"] = Voucher(
            name="Illusion",
            description="Playing cards in shop may have enhancements, editions, and seals",
            cost=10,
            effect=lambda state: {"enhanced_cards_in_shop": True},
            tier=2
        )
        
        vouchers["Petroglyph"] = Voucher(
            name="Petroglyph",
            description="-1 Ante, earn discounts are halved",
            cost=10,
            effect=lambda state: {"ante_reduction": 1, "earn_multiplier": 0.5},
            tier=2
        )
        
        vouchers["Retcon"] = Voucher(
            name="Retcon",
            description="Reroll Boss Blind unlimited times, $0 cost",
            cost=10,
            effect=lambda state: {"boss_reroll": -1},  # -1 means unlimited
            tier=2
        )
        
        vouchers["Palette"] = Voucher(
            name="Palette",
            description="+1 hand size",
            cost=10,
            effect=lambda state: {"hand_size_add": 1},
            tier=2
        )
        
        return vouchers

class VoucherShop:
    """Manages voucher generation for shop."""
    
    def __init__(self):
        self.all_vouchers = VoucherEffects.create_all()
        self.purchased_vouchers = set()
    
    def generate_shop_voucher(self, ante: int = 1) -> Optional[Voucher]:
        """Generate a voucher for the shop."""
        # Filter available vouchers
        available = []
        
        for name, voucher in self.all_vouchers.items():
            if name in self.purchased_vouchers:
                continue
                
            # Check if prerequisite purchased for tier 2
            if voucher.tier == 2:
                # Find the tier 1 version
                prereq = next((v for v in self.all_vouchers.values() 
                             if v.upgrade == name), None)
                if not prereq or prereq.name not in self.purchased_vouchers:
                    continue
            
            available.append(voucher)
        
        if not available:
            return None
        
        # Weight selection based on ante
        if ante >= 4:
            # Higher chance of tier 2 vouchers in later antes
            tier_weights = {1: 1, 2: 2}
        else:
            tier_weights = {1: 3, 2: 1}
        
        # Weighted random selection
        weighted_vouchers = []
        for voucher in available:
            weight = tier_weights.get(voucher.tier, 1)
            weighted_vouchers.extend([voucher] * weight)
        
        return random.choice(weighted_vouchers) if weighted_vouchers else None
    
    def purchase_voucher(self, voucher: Voucher):
        """Mark voucher as purchased."""
        self.purchased_vouchers.add(voucher.name)
