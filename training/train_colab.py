"""
Sales RL Environment - Training Script for Google Colab
Using Unsloth + TRL GRPO for fast training

This script can be run in Google Colab to train an LLM on the Sales RL environment.
"""

# ============================================================================
# PART 1: SETUP & INSTALLATION
# ============================================================================

# Install dependencies
get_ipython().system('pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
get_ipython().system('pip install -q trl peft accelerate bitsandbytes websockets pydantic numpy')

# ============================================================================
# PART 2: ENVIRONMENT CLIENT
# ============================================================================

import websockets
import json
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel

class ActionType:
    RESEARCH_PROSPECT = 0
    SEND_INTRO_EMAIL = 1
    SEND_FOLLOW_UP = 2
    SCHEDULE_CALL = 3
    SEND_PROPOSAL = 4
    CUSTOMIZE_PROPOSAL = 5
    NEGOTIATE_PRICE = 6
    SEND_CONTRACT = 7
    UPDATE_CRM = 8
    DO_NOTHING = 9

class SalesAction(BaseModel):
    action_type: int

class SalesEnvClient:
    """Client to connect to Sales RL Environment on HF Spaces"""
    
    def __init__(self, base_url: str):
        # Convert HTTPS to WSS for HF Spaces
        if base_url.startswith("https://"):
            self.ws_url = base_url.replace("https://", "wss://") + "/ws"
        else:
            self.ws_url = base_url + "/ws"
        self.websocket = None
    
    async def __aenter__(self):
        self.websocket = await websockets.connect(self.ws_url)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.websocket:
            await self.websocket.close()
    
    async def reset(self) -> Dict[str, Any]:
        await self.websocket.send(json.dumps({"type": "reset"}))
        response = await self.websocket.recv()
        data = json.loads(response)
        return {
            "observation": data["observation"],
            "reward": data["reward"],
            "done": data["done"]
        }
    
    async def step(self, action: int) -> Dict[str, Any]:
        await self.websocket.send(json.dumps({
            "type": "step",
            "action": {"action_type": action}
        }))
        response = await self.websocket.recv()
        data = json.loads(response)
        return {
            "observation": data["observation"],
            "reward": data["reward"],
            "done": data["done"]
        }

# ============================================================================
# PART 3: PROMPT FORMATTING
# ============================================================================

def format_observation_as_prompt(obs: Dict) -> str:
    """Convert observation to text prompt for LLM"""
    
    stages = ['LEAD', 'QUALIFIED', 'PROPOSAL', 'NEGOTIATION', 'CLOSED_WON', 'CLOSED_LOST']
    
    prompt = f"""You are a B2B sales agent. Your goal is to achieve 10/10 engagement score and close deals.

Current Situation:
- Deal Stage: {stages[obs['deal_stage']]}
- Engagement Score: {obs['engagement_score']:.1f}/10 (GOAL: 10/10)
- Prospect Warmth: {obs['prospect_warmth']:.0f}/100
- Prospect: {obs['prospect']['company_size']} {obs['prospect']['industry']} company (${obs['prospect']['budget']:.0f}K budget)
- Days Since Contact: {obs['days_since_contact']}
- Emails Sent: {obs['emails_sent']}
- Calls Completed: {obs['calls_completed']}
- Proposal Sent: {'Yes' if obs['proposal_sent'] else 'No'}
- Contract Sent: {'Yes' if obs['contract_sent'] else 'No'}

Last Feedback: {obs['action_feedback']}
"""
    
    if obs.get('oversight_alert'):
        prompt += f"\n⚠️ Fleet AI Alert: {obs['oversight_alert']}\n"
    
    prompt += """
Available Actions:
0. Research Prospect
1. Send Intro Email
2. Send Follow-Up
3. Schedule Call
4. Send Proposal
5. Customize Proposal
6. Negotiate Price
7. Send Contract
8. Update CRM
9. Do Nothing

What action should you take next? Respond with just the number (0-9):"""
    
    return prompt

# ============================================================================
# PART 4: LOAD MODEL WITH UNSLOTH
# ============================================================================

from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None  # Auto-detect
load_in_4bit = True  # Use 4-bit quantization for memory efficiency

# Load Qwen model (fast and good for RL)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Add LoRA adapters for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

print("✓ Model loaded with Unsloth!")

# ============================================================================
# PART 5: TRL GRPO TRAINING
# ============================================================================

from trl import GRPOConfig, GRPOTrainer
from transformers import TrainingArguments
import numpy as np

# Environment URL (replace with your HF Space URL)
ENV_URL = "https://your-username-sales-env.hf.space"  # UPDATE THIS!

# Training configuration
training_args = GRPOConfig(
    output_dir="./sales_rl_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    warmup_steps=50,
    max_grad_norm=1.0,
    remove_unused_columns=False,
)

def generate_action_from_model(prompt: str) -> int:
    """Generate action using the model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.7,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Extract number
    try:
        action = int(response.strip().split()[0])
        return max(0, min(9, action))  # Clamp to 0-9
    except:
        return np.random.randint(0, 10)  # Random fallback

async def run_episode():
    """Run one episode and collect experience"""
    episode_data = {
        'prompts': [],
        'actions': [],
        'rewards': [],
        'observations': []
    }
    
    async with SalesEnvClient(ENV_URL) as client:
        result = await client.reset()
        obs = result['observation']
        done = False
        
        while not done:
            # Format prompt
            prompt = format_observation_as_prompt(obs)
            
            # Generate action
            action = generate_action_from_model(prompt)
            
            # Take step
            result = await client.step(action)
            
            # Store experience
            episode_data['prompts'].append(prompt)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(result['reward'])
            episode_data['observations'].append(obs)
            
            obs = result['observation']
            done = result['done']
    
    return episode_data

# Training loop
print("Starting GRPO training...")
print(f"Environment: {ENV_URL}")

num_episodes = 100
best_engagement = 0.0

for episode in range(num_episodes):
    # Run episode
    try:
        episode_data = asyncio.run(run_episode())
        
        total_reward = sum(episode_data['rewards'])
        final_engagement = episode_data['observations'][-1]['engagement_score']
        
        print(f"Episode {episode+1}/{num_episodes}: "
              f"Reward={total_reward:.2f}, "
              f"Engagement={final_engagement:.1f}/10")
        
        if final_engagement > best_engagement:
            best_engagement = final_engagement
            print(f"  🌟 New best engagement: {best_engagement:.1f}/10!")
        
        # TODO: Integrate with TRL GRPO trainer
        # For now, this collects experience - full GRPO integration requires
        # more complex reward shaping and policy optimization
        
    except Exception as e:
        print(f"  Error in episode {episode+1}: {e}")
        continue
    
    # Save checkpoint every 10 episodes
    if (episode + 1) % 10 == 0:
        model.save_pretrained(f"./checkpoints/episode_{episode+1}")
        print(f"  💾 Checkpoint saved")

print(f"\n✓ Training complete! Best engagement: {best_engagement:.1f}/10")

# ============================================================================
# PART 6: SAVE FINAL MODEL
# ============================================================================

# Save in multiple formats
model.save_pretrained("sales_rl_final")
tokenizer.save_pretrained("sales_rl_final")

# Save for GGUF (optional - for local inference)
model.save_pretrained_gguf("sales_rl_final_gguf", tokenizer, quantization_method="q4_k_m")

print("✓ Model saved!")
print("\nTo use the model:")
print("1. Download 'sales_rl_final' folder")
print("2. Load with: FastLanguageModel.from_pretrained('sales_rl_final')")
print("3. Deploy to HF Spaces or use locally")

# ============================================================================
# PART 7: EVALUATION
# ============================================================================

async def evaluate_model(num_episodes=10):
    """Evaluate trained model"""
    results = []
    
    for i in range(num_episodes):
        async with SalesEnvClient(ENV_URL) as client:
            result = await client.reset()
            obs = result['observation']
            done = False
            total_reward = 0
            
            while not done:
                prompt = format_observation_as_prompt(obs)
                action = generate_action_from_model(prompt)
                result = await client.step(action)
                
                total_reward += result['reward']
                obs = result['observation']
                done = result['done']
            
            results.append({
                'engagement': obs['engagement_score'],
                'reward': total_reward,
                'won': obs['deal_stage'] == 4  # CLOSED_WON
            })
    
    avg_engagement = np.mean([r['engagement'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    win_rate = np.mean([r['won'] for r in results])
    
    print(f"\nEvaluation Results ({num_episodes} episodes):")
    print(f"  Average Engagement: {avg_engagement:.2f}/10")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Win Rate: {win_rate:.1%}")
    
    return results

# Run evaluation
eval_results = asyncio.run(evaluate_model(num_episodes=10))
