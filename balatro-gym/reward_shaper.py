class RewardShaper:
    """Shapes rewards to encourage good play patterns."""
    
    def __init__(self):
        # Reward weights
        self.weights = {
            "hand_score": 0.01,      # Small reward per point scored
            "blind_completion": 100,  # Large reward for beating blind
            "ante_progression": 200,  # Bonus for reaching new ante
            "money_saved": 0.1,      # Small reward for efficient money use
            "joker_synergy": 10,     # Bonus for good joker combinations
            "efficiency": 5,         # Bonus for beating blind with hands remaining
        }
    
    def calculate_hand_reward(self, hand_score: int, current_score: int, 
                            target_score: int) -> float:
        """Calculate reward for playing a hand."""
        reward = 0.0
        
        # Base reward for scoring
        reward += hand_score * self.weights["hand_score"]
        
        # Progress bonus (getting closer to target)
        progress = current_score / target_score
        if progress > 0.5:
            reward += 5.0  # Bonus for being over halfway
        if progress > 0.8:
            reward += 10.0  # Additional bonus for being close
        
        return reward
    
    def calculate_blind_reward(self, final_score: int, target_score: int, 
                             ante: int) -> float:
        """Calculate reward for completing a blind."""
        reward = 0.0
        
        if final_score >= target_score:
            # Base completion reward
            reward += self.weights["blind_completion"]
            
            # Efficiency bonus (overkill percentage)
            overkill = (final_score - target_score) / target_score
            reward += min(overkill * 20, 50)  # Cap at 50
            
            # Ante scaling
            reward *= (1 + ante * 0.1)  # 10% bonus per ante
        else:
            # Penalty for failing
            reward -= 50
            
            # Partial credit for getting close
            completion = final_score / target_score
            reward += completion * 20
        
        return reward
    
    def calculate_shop_reward(self, action: str, cost: int, money_before: int,
                            money_after: int) -> float:
        """Calculate reward for shop purchases."""
        reward = 0.0
        
        # Penalty for overspending
        if money_after < 0:
            return -100.0
        
        # Rewards based on purchase type
        purchase_values = {
            "joker": 5.0,
            "tarot": 2.0,
            "planet": 2.0,
            "spectral": 3.0,
            "voucher": 10.0,
        }
        
        for item_type, value in purchase_values.items():
            if item_type in action.lower():
                reward += value
                break
        
        # Efficiency bonus (saving money)
        if money_after > 10:
            reward += self.weights["money_saved"] * money_after
        
        return reward
    
    def calculate_episode_bonus(self, final_ante: int, total_money: int,
                              victory: bool) -> float:
        """Calculate final bonus rewards for the episode."""
        reward = 0.
