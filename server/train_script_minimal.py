"""
MINIMAL TRAINING SCRIPT - Sales RL Environment
Maintained across all iterations for consistency.

This script trains an LLM to optimize for 10/10 engagement scores.
Can be run in Google Colab or locally.
"""

import asyncio
import json
import numpy as np
from typing import List, Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

# UPDATE THIS to your deployed HF Space URL
ENV_URL = "wss://your-username-sales-env.hf.space/ws"  # or "ws://localhost:8000/ws" for local

# Training hyperparameters
NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 50
LEARNING_RATE = 1e-4

# ============================================================================
# ENVIRONMENT CLIENT
# ============================================================================

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'websockets'])
    import websockets

class SalesEnvClient:
    """Minimal client to connect to Sales RL Environment"""
    
    def __init__(self, url: str):
        self.url = url
        self.ws = None
    
    async def connect(self):
        """Connect to environment"""
        self.ws = await websockets.connect(self.url)
        print(f"✓ Connected to {self.url}")
    
    async def close(self):
        """Close connection"""
        if self.ws:
            await self.ws.close()
    
    async def reset(self) -> Dict:
        """Reset environment"""
        await self.ws.send(json.dumps({"type": "reset"}))
        response = json.loads(await self.ws.recv())
        return response
    
    async def step(self, action: int) -> Dict:
        """Take action"""
        await self.ws.send(json.dumps({
            "type": "step",
            "action": {"action_type": action}
        }))
        response = json.loads(await self.ws.recv())
        return response

# ============================================================================
# SIMPLE POLICY (to be replaced with LLM)
# ============================================================================

class SimplePolicy:
    """Simple rule-based policy for baseline"""
    
    def __init__(self):
        self.episode_count = 0
    
    def select_action(self, obs: Dict, step: int) -> int:
        """Select action based on state"""
        stage = obs['deal_stage']
        warmth = obs['prospect_warmth']
        emails = obs['emails_sent']
        calls = obs['calls_completed']
        proposal_sent = obs['proposal_sent']
        contract_sent = obs['contract_sent']
        
        # Rule-based logic
        if stage == 0:  # LEAD
            if emails == 0:
                return 1  # Send Intro Email
            elif warmth < 60:
                return 0  # Research Prospect
            else:
                return 2  # Send Follow-Up
        
        elif stage == 1:  # QUALIFIED
            if calls == 0:
                return 3  # Schedule Call
            elif warmth < 70:
                return 2  # Follow Up
            else:
                return 8  # Update CRM
        
        elif stage == 2:  # PROPOSAL
            if not proposal_sent:
                return 4  # Send Proposal
            elif warmth < 80:
                return 5  # Customize Proposal
            else:
                return 8  # Update CRM
        
        elif stage == 3:  # NEGOTIATION
            if not contract_sent:
                if warmth < 75:
                    return 6  # Negotiate Price
                else:
                    return 7  # Send Contract
            else:
                return 8  # Update CRM
        
        return 9  # Do Nothing
    
    def learn(self, episode_data: Dict):
        """Learn from episode (placeholder for RL update)"""
        self.episode_count += 1
        engagement = episode_data['final_engagement']
        reward = episode_data['total_reward']
        won = episode_data['won']
        
        print(f"  Episode {self.episode_count}: "
              f"Engagement={engagement:.1f}/10, "
              f"Reward={reward:.1f}, "
              f"{'WON' if won else 'LOST'}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

async def train():
    """Main training loop"""
    
    print("="*60)
    print("SALES RL TRAINING")
    print("="*60)
    print(f"Environment: {ENV_URL}")
    print(f"Episodes: {NUM_EPISODES}")
    print("="*60 + "\n")
    
    # Create client and policy
    client = SalesEnvClient(ENV_URL)
    await client.connect()
    
    policy = SimplePolicy()
    
    # Training metrics
    all_rewards = []
    all_engagements = []
    wins = 0
    
    try:
        for episode in range(NUM_EPISODES):
            # Reset environment
            result = await client.reset()
            obs = result['observation']
            done = False
            step = 0
            
            episode_rewards = []
            
            # Run episode
            while not done and step < MAX_STEPS_PER_EPISODE:
                # Select action
                action = policy.select_action(obs, step)
                
                # Take step
                result = await client.step(action)
                obs = result['observation']
                reward = result['reward']
                done = result['done']
                
                episode_rewards.append(reward)
                step += 1
            
            # Episode complete
            total_reward = sum(episode_rewards)
            final_engagement = obs['engagement_score']
            won = obs['deal_stage'] == 4  # CLOSED_WON
            
            if won:
                wins += 1
            
            # Store metrics
            all_rewards.append(total_reward)
            all_engagements.append(final_engagement)
            
            # Learn from episode
            episode_data = {
                'total_reward': total_reward,
                'final_engagement': final_engagement,
                'won': won,
                'steps': step
            }
            policy.learn(episode_data)
            
            # Print progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(all_rewards[-10:])
                avg_engagement = np.mean(all_engagements[-10:])
                win_rate = wins / (episode + 1)
                
                print(f"\n--- After {episode + 1} episodes ---")
                print(f"Avg Reward (last 10): {avg_reward:.2f}")
                print(f"Avg Engagement (last 10): {avg_engagement:.2f}/10")
                print(f"Win Rate: {win_rate:.1%}\n")
    
    finally:
        await client.close()
    
    # Final results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total Episodes: {NUM_EPISODES}")
    print(f"Final Win Rate: {wins/NUM_EPISODES:.1%}")
    print(f"Avg Engagement: {np.mean(all_engagements):.2f}/10")
    print(f"Best Engagement: {max(all_engagements):.2f}/10")
    print(f"Avg Reward: {np.mean(all_rewards):.2f}")
    print("="*60)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    
    if IN_COLAB:
        print("Running in Google Colab")
    
    # Run training
    asyncio.run(train())

# ============================================================================
# NEXT STEPS: Replace SimplePolicy with LLM
# ============================================================================

"""
To integrate with Unsloth/TRL:

1. Load model:
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct")

2. Replace select_action with LLM call:
   def select_action(self, obs, step):
       prompt = format_obs_to_prompt(obs)
       output = model.generate(prompt)
       action = parse_output_to_action(output)
       return action

3. Add RL updates:
   def learn(self, episode_data):
       # Use TRL GRPO to update model based on rewards
       # Optimize for engagement_score + total_reward
       pass

See training/train_colab.py for full Unsloth+TRL implementation.
"""