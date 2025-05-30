import numpy as np
import json
import re
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent

class LLMBalatroAgent(BaseAgent):
    """LLM-based agent for playing Balatro."""
    
    def __init__(self, llm_client, name: str = "LLMAgent", 
                 temperature: float = 0.7, use_cot: bool = True):
        super().__init__(name)
        self.llm = llm_client
        self.temperature = temperature
        self.use_cot = use_cot  # Chain of thought reasoning
        
        # Prompt components
        self.system_prompt = self._create_system_prompt()
        self.few_shot_examples = self._create_few_shot_examples()
    
    def get_action(self, observation: Dict[str, np.ndarray], 
                   action_mask: np.ndarray, env) -> int:
        """Select action using LLM reasoning."""
        # Get state description
        state_desc = env.get_state_description()
        legal_actions = env.describe_legal_actions(action_mask)
        
        # Build prompt
        prompt = self._build_prompt(state_desc, legal_actions, env)
        
        # Get LLM response
        response = self._query_llm(prompt)
        
        # Parse action from response
        action_idx = self._parse_action(response, action_mask)
        
        return action_idx
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the LLM."""
        return """You are an expert Balatro player. Your goal is to progress through antes by scoring enough points to beat each blind (small, big, and boss).

Key strategies:
1. Build powerful poker hands by selecting the right cards
2. Manage your discards wisely - you only get 3 per blind
3. Buy jokers that synergize with your playstyle
4. Save money for important purchases but don't be too greedy
5. Know when to skip a blind if you can't beat it

You must respond with a single action number from the legal actions list."""
    
    def _create_few_shot_examples(self) -> List[Dict[str, str]]:
        """Create few-shot examples for better decision making."""
        return [
            {
                "state": "Ante 1, Small blind (300 target), 0 scored, 4 hands left",
                "actions": "[0] Play card 0, [1] Play card 1, [5] Discard hand",
                "reasoning": "With 4 hands to score 300 points, I should play cards to start scoring.",
                "action": "0"
            },
            {
                "state": "In shop with $15, Jokers: [Slot 0: +4 Mult - $6]",
                "actions": "[9] Buy joker slot 0, [14] Buy tarot pack",
                "reasoning": "+4 Mult joker is excellent early game value. Worth buying over packs.",
                "action": "9"
            }
        ]
    
    def _build_prompt(self, state: str, actions: str, env) -> str:
        """Build the full prompt for the LLM."""
        prompt_parts = []
        
        # Add few-shot examples if available
        if self.few_shot_examples:
            prompt_parts.append("Here are some example decisions:\n")
            for example in self.few_shot_examples:
                prompt_parts.append(f"State: {example['state']}")
                prompt_parts.append(f"Legal actions: {example['actions']}")
                if self.use_cot:
                    prompt_parts.append(f"Reasoning: {example['reasoning']}")
                prompt_parts.append(f"Action chosen: {example['action']}\n")
        
        # Current state
        prompt_parts.append("Current game state:")
        prompt_parts.append(state)
        prompt_parts.append("\n" + actions)
        
        if self.use_cot:
            prompt_parts.append("\nThink step by step about the best action, then provide your choice.")
            prompt_parts.append("Format: First explain your reasoning, then write 'Action: [number]'")
        else:
            prompt_parts.append("\nChoose the best action number:")
        
        return "\n".join(prompt_parts)
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with the prompt."""
        # This is a placeholder - adapt to your specific LLM API
        # Example for OpenAI API:
        try:
            response = self.llm.complete(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM query failed: {e}")
            return "18"  # Return NO_OP as fallback
    
    def _parse_action(self, response: str, action_mask: np.ndarray) -> int:
        """Parse action index from LLM response."""
        # Try to find action number in response
        # Look for patterns like "Action: 5" or just "5"
        patterns = [
            r"Action:\s*(\d+)",
            r"action:\s*(\d+)",
            r"\[(\d+)\]",
            r"^(\d+)$"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.MULTILINE)
            if matches:
                try:
                    action_idx = int(matches[-1])  # Take last match
                    # Verify it's legal
                    if 0 <= action_idx < len(action_mask) and action_mask[action_idx]:
                        return action_idx
                except ValueError:
                    continue
        
        # Fallback: return first legal action
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) > 0:
            print(f"Failed to parse action from: {response}")
            return legal_actions[0]
        
        return 18  # NO_OP
