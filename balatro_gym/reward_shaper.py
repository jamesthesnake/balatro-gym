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
            "penalty_no_op": -0.1, # Example: small penalty for doing nothing
            "reward_discard": 0.05, # Example: small reward for strategic discard

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
        reward = 0.0
        
        # Victory bonus
        if victory:
            reward += 1000.0
        
        # Ante progression bonus
        reward += (final_ante - 1) * self.weights["ante_progression"]
        
        # Money management bonus
        if total_money > 50:
            reward += total_money * self.weights["money_saved"]
        
        return reward

    def on_play(self, played_card_info, env_state) -> float:
        """Reward for the action of playing a card/hand."""
        # This method is called *after* a card is popped and the hand is evaluated in _apply_action.
        # The 'played_card_info' is the single card popped in your current env.py.
        # The actual score from the hand evaluation (chips * mult) is already calculated
        # and added to env_state.current_score in _apply_action *before* this would be called
        # if on_play was called *after* score calculation.
        # Your current _apply_action calls:
        #   reward += self.reward_shaper.on_play(card, self)
        # where 'card' is the popped card.
        # The calculated_score is NOT directly passed to this on_play.
        # You could decide to give a small reward for the act of playing, or use calculate_hand_reward
        # if you restructure _apply_action to pass the hand_score here.
        # For now, a simple placeholder:
        # print(f"on_play called. Card: {played_card_info}, Current Score: {env_state.current_score}") # For debugging
        return 0.0 # Placeholder: Define reward logic

    def on_discard(self, env_state) -> float:
        """Reward/penalty for discarding."""
        # print("on_discard called") # For debugging
        return self.weights.get("reward_discard", 0.05) # Example

    def on_end_hand(self, env_state) -> float:
        """Reward for ending a hand (playing it out against the blind)."""
        # This is called when Action.END_HAND is chosen.
        # Your _apply_action for END_HAND currently does:
        #   reward += self.reward_shaper.on_end_hand(self)
        #   self.hands_remaining = 0
        # It does not itself evaluate a score. Score evaluation happens on PLAY_CARD_X.
        # This on_end_hand might be a structural reward for completing a hand attempt.
        # Or it might be the point where you check if the blind was beaten.
        # print("on_end_hand called") # For debugging
        # if env_state.current_score >= env_state.target_score:
        #     return self.calculate_blind_reward(env_state.current_score, env_state.target_score, env_state.current_ante)
        # else:
        #     # Maybe a small penalty if hands_remaining became 0 but target not met
        #     if env_state.hands_remaining == 0: # after the decrement
        #          return -5.0
        return 0.0 # Placeholder: Define reward logic, possibly linking to blind completion

    def on_skip_blind(self, env_state) -> float:
        """Reward/penalty for skipping a blind."""
        # Your _apply_action for SKIP_BLIND already has:
        #   cost = self._skip_cost()
        #   self.money -= cost
        #   reward -= cost * 0.1  <-- Direct penalty based on cost
        # So this on_skip_blind might be for additional shaping or could return 0 if already handled.
        # print("on_skip_blind called") # For debugging
        return 0.0 # Already handled in _apply_action

    def on_use_consumable(self, consumable_used_info, env_state) -> float:
        """Reward for using a consumable."""
        # 'consumable_used_info' is the consumable object that was popped.
        # print(f"on_use_consumable called. Consumable: {consumable_used_info}") # For debugging
        # You could have logic here based on consumable_used_info.type or .name
        return 1.0 # Example: Small flat reward for using any consumable

    def on_buy_joker(self, joker_bought_info, env_state) -> float:
        """Reward for buying a joker."""
        # 'joker_bought_info' is the joker object.
        # print(f"on_buy_joker called. Joker: {joker_bought_info}") # For debugging
        # You could use self.calculate_shop_reward here if you adapt its parameters
        # or give a fixed/joker-dependent reward.
        # The action name for calculate_shop_reward is tricky here, as it's generic.
        # money_before would need to be passed or calculated.
        return self.calculate_shop_reward(action="buy_joker", cost=joker_bought_info.cost,
                                           money_before=env_state.money + joker_bought_info.cost, # Approx
                                           money_after=env_state.money)


    def on_buy_pack(self, pack_type_bought, env_state) -> float:
        """Reward for buying a pack."""
        # 'pack_type_bought' is a string like "tarot", "planet".
        # print(f"on_buy_pack called. Pack type: {pack_type_bought}") # For debugging
        pack_base_cost = 4
        effective_pack_cost = pack_base_cost * (1 - env_state.voucher_effects.get("shop_discount", 0))
        return self.calculate_shop_reward(action=f"buy_pack_{pack_type_bought}", cost=int(effective_pack_cost),
                                           money_before=env_state.money + int(effective_pack__cost), # Approx
                                           money_after=env_state.money)

    def on_buy_voucher(self, voucher_bought_info, env_state) -> float:
        """Reward for buying a voucher."""
        # 'voucher_bought_info' is the voucher object.
        # print(f"on_buy_voucher called. Voucher: {voucher_bought_info}") # For debugging
        return self.calculate_shop_reward(action="buy_voucher", cost=voucher_bought_info.cost,
                                           money_before=env_state.money + voucher_bought_info.cost, # Approx
                                           money_after=env_state.money)

    def on_no_op(self, env_state) -> float:
        """Reward/penalty for NO_OP."""
        # print("on_no_op called") # For debugging
        # If in shop and NO_OP is the only option, maybe it's okay.
        # If in round with playable hands, NO_OP might be bad.
        return self.weights.get("penalty_no_op", -0.1) # Example: Small penalty by default

