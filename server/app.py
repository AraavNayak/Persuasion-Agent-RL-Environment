"""
FastAPI server with backend Q-learning agent and auto-playing web UI.
Shows sequential action execution with visual highlighting.

KEY IMPROVEMENT: Q-learning now happens in Python backend, learning directly
from the sophisticated reward function in sales_environment.py.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
from pathlib import Path
import json
import httpx
import asyncio
import os
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import SalesAction, ActionType
from server.sales_environment import SalesEnvironment, save_policy_hints

TAKEAWAY_FILE  = Path(__file__).parent / "sales_agent_takeaways.txt"
OVERSIGHT_FILE = Path(__file__).parent / "fleet_oversight_log.txt"
QTABLE_FILE    = Path(__file__).parent / "qtable.pkl"

episode_counter = 0
episode_history = []

# ============================================================================
# Backend Q-Learning Agent (the actual learning happens here now)
# ============================================================================

class QLearningAgent:
    """
    Tabular Q-learning agent that learns optimal sales policies.
    
    State space: (deal_stage, warmth_bucket, calls_bucket, proposal_sent)
    Action space: 10 sales actions
    
    This agent persists between episodes and actually learns from the
    sophisticated reward function in sales_environment.py.
    """
    
    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        
        # Q-table indexed by discretized state
        # State = (stage: 0-3, warmth: 0-4, calls: 0-3, proposal: 0-1)
        # Approx 4 * 5 * 4 * 2 = 160 states
        self.Q = {}
        
        # Hyperparameters
        self.alpha = 0.15      # Learning rate (lowered for stability)
        self.gamma = 0.95      # Discount factor (high for long episodes)
        self.epsilon = 0.5     # Exploration rate (starts higher)
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.96  # Decay per episode
        
        # Training stats
        self.episode_count = 0
        self.last_state = None
        self.last_action = None
        
        # Policy hints integration
        self.policy_hints = {}
        
    def discretize_state(self, obs: dict) -> tuple:
        """Convert continuous observation to discrete state tuple."""
        stage = min(3, max(0, obs.get('deal_stage', 0)))
        
        # Warmth buckets: 0-20, 20-40, 40-60, 60-80, 80-100
        warmth = min(4, int(obs.get('prospect_warmth', 50) / 20))
        
        # Calls buckets: 0, 1, 2, 3+
        calls = min(3, obs.get('calls_completed', 0))
        
        # Proposal sent: boolean
        proposal = 1 if obs.get('proposal_sent', False) else 0
        
        return (stage, warmth, calls, proposal)
    
    def get_q_value(self, state: tuple, action: int) -> float:
        """Get Q-value for state-action pair, initializing if needed."""
        if state not in self.Q:
            # Initialize with policy hints if available
            self.Q[state] = self._initialize_q_row(state)
        return self.Q[state][action]
    
    def _initialize_q_row(self, state: tuple) -> list:
        """Initialize Q-values for a new state, optionally using policy hints."""
        q_row = [0.0] * self.n_actions
        
        # Apply policy hints if available
        if self.policy_hints and 'action_bias' in self.policy_hints:
            stage, warmth, calls, proposal = state
            stage_names = ['LEAD', 'QUALIFIED', 'PROPOSAL', 'NEGOTIATION']
            
            if stage < len(stage_names):
                stage_name = stage_names[stage]
                if stage_name in self.policy_hints['action_bias']:
                    action_names = [
                        'RESEARCH_PROSPECT', 'SEND_INTRO_EMAIL', 'SEND_FOLLOW_UP',
                        'SCHEDULE_CALL', 'SEND_PROPOSAL', 'CUSTOMIZE_PROPOSAL',
                        'NEGOTIATE_PRICE', 'SEND_CONTRACT', 'UPDATE_CRM', 'DO_NOTHING'
                    ]
                    
                    for action_idx, action_name in enumerate(action_names):
                        bias = self.policy_hints['action_bias'][stage_name].get(action_name, 0)
                        q_row[action_idx] = bias * 0.3  # Scale down the hints
        
        return q_row
    
    def choose_action(self, obs: dict) -> int:
        """Epsilon-greedy action selection."""
        state = self.discretize_state(obs)
        
        # Exploration
        if np.random.random() < self.epsilon:
            # Smart exploration: avoid clearly bad actions
            avoid_actions = self.policy_hints.get('avoid_actions', [])
            action_names = [
                'RESEARCH_PROSPECT', 'SEND_INTRO_EMAIL', 'SEND_FOLLOW_UP',
                'SCHEDULE_CALL', 'SEND_PROPOSAL', 'CUSTOMIZE_PROPOSAL',
                'NEGOTIATE_PRICE', 'SEND_CONTRACT', 'UPDATE_CRM', 'DO_NOTHING'
            ]
            
            weights = [
                0.1 if action_names[i] in avoid_actions else 1.0
                for i in range(self.n_actions)
            ]
            weights_sum = sum(weights)
            probabilities = [w / weights_sum for w in weights]
            
            action = np.random.choice(self.n_actions, p=probabilities)
        else:
            # Exploitation: choose best action
            q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            action = np.random.choice(best_actions)
        
        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, reward: float, next_obs: dict, done: bool):
        """Q-learning update rule."""
        if self.last_state is None or self.last_action is None:
            return
        
        next_state = self.discretize_state(next_obs)
        
        # Q-learning: Q(s,a) += α[r + γ max_a' Q(s',a') - Q(s,a)]
        current_q = self.get_q_value(self.last_state, self.last_action)
        
        if done:
            target = reward
        else:
            next_q_values = [self.get_q_value(next_state, a) for a in range(self.n_actions)]
            max_next_q = max(next_q_values)
            target = reward + self.gamma * max_next_q
        
        # Update Q-table
        td_error = target - current_q
        new_q = current_q + self.alpha * td_error
        
        # Ensure state exists in Q-table
        if self.last_state not in self.Q:
            self.Q[self.last_state] = self._initialize_q_row(self.last_state)
        
        self.Q[self.last_state][self.last_action] = new_q
    
    def end_episode(self):
        """Called at episode end to decay epsilon and reset episode state."""
        self.episode_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.last_state = None
        self.last_action = None
    
    def set_policy_hints(self, hints: dict):
        """Update policy hints from oversight agent."""
        self.policy_hints = hints
    
    def get_policy_summary(self) -> dict:
        """Get learned policy for each stage (for visualization)."""
        action_names = [
            'Research', 'Intro Email', 'Follow-Up', 'Call',
            'Proposal', 'Customize', 'Negotiate', 'Contract',
            'Update CRM', 'Do Nothing'
        ]
        stage_names = ['LEAD', 'QUALIFIED', 'PROPOSAL', 'NEGOTIATION']
        
        policy = {}
        for stage_idx, stage_name in enumerate(stage_names):
            # Find representative state for this stage
            # Use middle warmth (2), some calls (1), no proposal yet
            state = (stage_idx, 2, 1, 0)
            
            if state in self.Q:
                q_values = self.Q[state]
                best_action = int(np.argmax(q_values))
                max_q = float(q_values[best_action])
            else:
                # Initialize and check
                self._initialize_q_row(state)
                q_values = self.Q.get(state, [0.0] * self.n_actions)
                best_action = int(np.argmax(q_values))
                max_q = float(q_values[best_action])
            
            policy[stage_name] = {
                'action': action_names[best_action],
                'q_value': max_q
            }
        
        return policy
    
    def save(self, filepath: Path):
        """Persist Q-table to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'Q': self.Q,
                'epsilon': self.epsilon,
                'episode_count': self.episode_count,
            }, f)
    
    def load(self, filepath: Path):
        """Load Q-table from disk."""
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.Q = data.get('Q', {})
                self.epsilon = data.get('epsilon', self.epsilon)
                self.episode_count = data.get('episode_count', 0)

# Global agent instance
q_agent = QLearningAgent()

# Try to load existing Q-table
if QTABLE_FILE.exists():
    try:
        q_agent.load(QTABLE_FILE)
        print(f"[Q-agent] Loaded Q-table with {len(q_agent.Q)} states, epsilon={q_agent.epsilon:.3f}", flush=True)
    except Exception as e:
        print(f"[Q-agent] Failed to load Q-table: {e}", flush=True)


# ============================================================================
# Utility functions (unchanged)
# ============================================================================

def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


async def call_claude(prompt: str, max_tokens: int = 400) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        })
        return resp.json()["response"].strip()


def log_takeaway(action: SalesAction, result, episode_steps: list):
    obs        = _get(result, "observation", {})
    reward     = _get(result, "reward", 0)
    engagement = _get(obs, "engagement_score", 0)
    feedback   = _get(obs, "action_feedback", "")
    episode_steps.append({
        "action":     str(action.action_type).replace("ActionType.", ""),
        "reward":     round(float(reward), 3),
        "engagement": round(float(engagement), 2),
        "feedback":   feedback,
    })


async def write_episode_summary(result, episode_steps: list):
    global episode_counter
    episode_counter += 1
    episode_steps = list(episode_steps)

    state = _get(result, "state", {})
    obs   = _get(result, "observation", {})

    total_reward = float(_get(state, "total_reward", 0))
    engagement   = float(_get(obs, "engagement_score", 0))
    step_count   = int(_get(state, "step_count", len(episode_steps)))
    deal_stages  = ['LEAD', 'QUALIFIED', 'PROPOSAL', 'NEGOTIATION', 'CLOSED_WON', 'CLOSED_LOST']
    deal_idx     = int(_get(obs, "deal_stage", 0))
    deal_stage   = deal_stages[deal_idx] if 0 <= deal_idx < len(deal_stages) else "UNKNOWN"
    outcome      = "WON" if deal_stage == "CLOSED_WON" else "LOST"

    prospect     = _get(obs, "prospect", {}) or {}
    prospect_str = (
        f"size={prospect.get('company_size','?')}  "
        f"industry={prospect.get('industry','?')}  "
        f"decision_speed={prospect.get('decision_speed','?')}  "
        f"price_sensitivity={prospect.get('price_sensitivity',0):.2f}  "
        f"relationship_importance={prospect.get('relationship_importance',0):.2f}"
    )

    episode_history.append({
        "episode":      episode_counter,
        "outcome":      outcome,
        "deal_stage":   deal_stage,
        "total_reward": total_reward,
        "engagement":   engagement,
        "step_count":   step_count,
        "steps":        episode_steps,
        "prospect":     prospect,
    })

    prior_context = ""
    if TAKEAWAY_FILE.exists():
        prior_text = TAKEAWAY_FILE.read_text().strip()
        if prior_text:
            prior_context = prior_text[-1500:]

    step_lines = []
    for i, s in enumerate(episode_steps):
        step_lines.append(
            f"  Step {i+1}: {s['action']} | reward={s['reward']:+.2f} | "
            f"engagement={s['engagement']:.1f} | feedback: {s['feedback']}"
        )
    steps_text = "\n".join(step_lines) if step_lines else "  (no steps recorded)"

    prompt = f"""You are an RL training analyst watching a sales agent learn over many episodes.

PREVIOUS EPISODE TAKEAWAYS (for context on what changed):
{prior_context if prior_context else "(no prior episodes yet)"}

EPISODE {episode_counter} RESULTS:
- Outcome: {outcome}
- Final deal stage: {deal_stage}
- Total reward: {total_reward:.2f}
- Final engagement score: {engagement:.1f}/10
- Steps taken: {step_count}/50
- Prospect profile: {prospect_str}

STEP-BY-STEP BREAKDOWN:
{steps_text}

Write 3-5 concise bullet point takeaways for this episode. Be specific about:
- Which exact actions helped or hurt given THIS prospect's personality
- How prospect traits (price_sensitivity, relationship_importance, decision_speed) affected what worked
- How this episode differed from prior ones
- What the agent should do differently next time for this prospect archetype

Be analytical and specific. Reference actual step data and prospect traits."""

    summary_lines = []
    try:
        llm_text = await call_claude(prompt, max_tokens=400)
        summary_lines = llm_text.splitlines()
    except Exception as e:
        print(f"[takeaways] LLM call failed: {e}", flush=True)
        summary_lines = [
            f"- Outcome: {outcome} | Reward: {total_reward:.2f} | Engagement: {engagement:.1f}/10",
            f"- Prospect: {prospect_str}",
        ]

    try:
        TAKEAWAY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TAKEAWAY_FILE, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Episode {episode_counter} | {deal_stage} | Reward: {total_reward:.2f} | "
                    f"Engagement: {engagement:.1f}/10 | Steps: {step_count}\n")
            f.write(f"Prospect: {prospect_str}\n")
            f.write(f"{'='*50}\n")
            for line in summary_lines:
                f.write(line + "\n")
            f.flush()
        print(f"[takeaways] wrote episode {episode_counter} summary", flush=True)
    except Exception as e:
        print(f"[takeaways] ERROR writing file: {e}", flush=True)

    asyncio.create_task(write_oversight_analysis())


async def write_oversight_analysis():
    if not episode_history:
        return

    table_lines = ["Episode | Outcome | Stage         | Reward  | Engagement | Steps | Prospect"]
    table_lines.append("-" * 90)
    for ep in episode_history[-20:]:
        p = ep.get("prospect", {})
        pstr = f"{p.get('company_size','?')}/{p.get('decision_speed','?')}/ri={p.get('relationship_importance',0):.1f}/ps={p.get('price_sensitivity',0):.1f}"
        table_lines.append(
            f"  {ep['episode']:>3}   | {ep['outcome']:<6}  | {ep['deal_stage']:<13} | "
            f"{ep['total_reward']:>6.2f}  | {ep['engagement']:>5.1f}/10    | "
            f"{ep['step_count']}/50 | {pstr}"
        )
    table = "\n".join(table_lines)

    latest = episode_history[-1]
    step_lines = []
    for i, s in enumerate(latest["steps"]):
        step_lines.append(
            f"  Step {i+1}: {s['action']} | reward={s['reward']:+.2f} | "
            f"engagement={s['engagement']:.1f} | feedback: {s['feedback']}"
        )
    latest_steps = "\n".join(step_lines) if step_lines else "  (no steps)"

    prior_oversight = ""
    if OVERSIGHT_FILE.exists():
        text = OVERSIGHT_FILE.read_text().strip()
        if text:
            prior_oversight = text[-2000:]

    total_episodes = len(episode_history)
    wins      = sum(1 for e in episode_history if e["outcome"] == "WON")
    losses    = total_episodes - wins
    win_rate  = wins / total_episodes * 100

    recent_rewards = [e["total_reward"] for e in episode_history[-5:]]
    reward_trend   = (
        "improving" if len(recent_rewards) > 1 and recent_rewards[-1] > recent_rewards[0] else
        "declining" if len(recent_rewards) > 1 and recent_rewards[-1] < recent_rewards[0] else
        "flat"
    )

    action_counts: dict = {}
    for ep in episode_history:
        for s in ep["steps"]:
            action_counts[s["action"]] = action_counts.get(s["action"], 0) + 1
    total_actions   = sum(action_counts.values()) or 1
    action_freq     = sorted(action_counts.items(), key=lambda x: -x[1])
    action_freq_str = ", ".join(
        f"{a}={c} ({c/total_actions*100:.0f}%)" for a, c in action_freq[:8]
    )

    prompt = f"""You are a Fleet AI Oversight Agent monitoring an RL sales agent across training episodes.

The reward function depends on prospect personality:
- High relationship_importance (ri) prospects reward SCHEDULE_CALL and CUSTOMIZE_PROPOSAL more
- High price_sensitivity (ps) prospects reward RESEARCH_PROSPECT more; discounts are expected
- Fast decision_speed prospects reward SEND_PROPOSAL and SEND_CONTRACT more
- Enterprise/slow prospects need more calls before advancing stages

=== CROSS-EPISODE PERFORMANCE TABLE (last 20 episodes) ===
{table}

=== OVERALL STATS ===
Total episodes: {total_episodes}
Win/Loss: {wins}W / {losses}L  ({win_rate:.1f}% win rate)
Reward trend (last 5 episodes): {reward_trend}
Action frequency across ALL episodes: {action_freq_str}

=== LATEST EPISODE ({latest['episode']}) STEP BREAKDOWN ===
{latest_steps}

=== YOUR PRIOR OVERSIGHT ANALYSIS (for continuity) ===
{prior_oversight if prior_oversight else "(no prior analysis)"}

Write a Fleet AI Oversight Report covering:

1. **BEHAVIOURAL TRENDS** — Is the agent adapting strategy to different prospect types?
2. **ACTION AUDIT** — Which actions is it over/under-using? Does usage match prospect personality?
3. **PERFORMANCE TRAJECTORY** — Genuine improvement, plateau, or regression? Reference numbers.
4. **RED FLAGS** — Reward hacking, stuck loops, ignoring prospect traits, excessive discounting?
5. **OVERSIGHT VERDICT** — 1-2 sentence assessment.
6. **POLICY HINTS** — CRITICAL: End your response with a JSON block (and nothing after it).
   Use ONLY these action names: RESEARCH_PROSPECT, SEND_INTRO_EMAIL, SEND_FOLLOW_UP,
   SCHEDULE_CALL, SEND_PROPOSAL, CUSTOMIZE_PROPOSAL, NEGOTIATE_PRICE, SEND_CONTRACT,
   UPDATE_CRM, DO_NOTHING.
   Use ONLY these state names: LEAD, QUALIFIED, PROPOSAL, NEGOTIATION.
   Values are additive Q-value seeds (-10 to +15).

```json
{{
  "action_bias": {{
    "LEAD":        {{"RESEARCH_PROSPECT": 6, "SEND_INTRO_EMAIL": 8, "DO_NOTHING": -8}},
    "QUALIFIED":   {{"SCHEDULE_CALL": 10, "SEND_FOLLOW_UP": 4, "DO_NOTHING": -6}},
    "PROPOSAL":    {{"CUSTOMIZE_PROPOSAL": 8, "SEND_PROPOSAL": 6, "DO_NOTHING": -5}},
    "NEGOTIATION": {{"SEND_CONTRACT": 12, "NEGOTIATE_PRICE": 3, "DO_NOTHING": -8}}
  }},
  "avoid_actions": ["DO_NOTHING"],
  "strategy_note": "Prioritise relationship-building early; customize for high-ri prospects"
}}
```"""

    try:
        analysis = await call_claude(prompt, max_tokens=800)
    except Exception as e:
        print(f"[oversight] LLM call failed: {e}", flush=True)
        analysis = f"[Oversight LLM call failed: {e}]"

    import re, json as _json
    hints_saved = False
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', analysis, re.DOTALL)
        if json_match:
            hints = _json.loads(json_match.group(1))
            save_policy_hints(hints)
            q_agent.set_policy_hints(hints)  # Update agent's policy hints
            hints_saved = True
            print(f"[oversight] policy hints saved: {list(hints.keys())}", flush=True)
        else:
            obj_match = re.search(r'\{[\s\S]*"action_bias"[\s\S]*\}', analysis)
            if obj_match:
                hints = _json.loads(obj_match.group(0))
                save_policy_hints(hints)
                q_agent.set_policy_hints(hints)
                hints_saved = True
    except Exception as e:
        print(f"[oversight] policy hints parse failed: {e}", flush=True)

    try:
        OVERSIGHT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OVERSIGHT_FILE, "a") as f:
            f.write(f"\n{'#'*60}\n")
            f.write(f"FLEET AI OVERSIGHT REPORT — After Episode {latest['episode']}\n")
            f.write(f"Win Rate: {win_rate:.1f}%  |  Reward Trend: {reward_trend}  |  Total Episodes: {total_episodes}\n")
            f.write(f"Policy hints saved: {hints_saved}\n")
            f.write(f"{'#'*60}\n\n")
            f.write(analysis + "\n")
            f.flush()
        print(f"[oversight] wrote analysis after episode {latest['episode']}", flush=True)
    except Exception as e:
        print(f"[oversight] ERROR writing file: {e}", flush=True)


environments = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    TAKEAWAY_FILE.parent.mkdir(parents=True, exist_ok=True)
    TAKEAWAY_FILE.write_text("")
    OVERSIGHT_FILE.write_text("")
    print("[startup] Cleared takeaways and oversight log files.", flush=True)
    print(f"[startup] Q-agent loaded with {len(q_agent.Q)} states", flush=True)
    yield
    # Save Q-table on shutdown
    q_agent.save(QTABLE_FILE)
    print(f"[shutdown] Saved Q-table with {len(q_agent.Q)} states", flush=True)
    environments.clear()

app = FastAPI(
    title="Sales RL Environment with Fleet AI Oversight",
    description="Auto-playing B2B Sales RL with oversight agent",
    version="0.5.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"name": "Sales RL Environment with Fleet AI Oversight", "version": "0.5.0", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/takeaways")
async def get_takeaways():
    if TAKEAWAY_FILE.exists():
        return {"content": TAKEAWAY_FILE.read_text()}
    return {"content": "No takeaways recorded yet."}

@app.get("/oversight")
async def get_oversight():
    if OVERSIGHT_FILE.exists():
        return {"content": OVERSIGHT_FILE.read_text()}
    return {"content": "No oversight analysis yet. Complete an episode first."}

@app.get("/agent_stats")
async def get_agent_stats():
    """Return Q-learning agent statistics."""
    return {
        "total_states": len(q_agent.Q),
        "epsilon": round(q_agent.epsilon, 3),
        "episodes_trained": q_agent.episode_count,
        "learned_policy": q_agent.get_policy_summary()
    }


@app.get("/policy_heatmap")
async def policy_heatmap():
    """Return Q-values for different prospect archetypes."""
    archetypes = {
        'Enterprise/Slow/High-RI': (3, 4, 2, 0),  # state tuple
        'Startup/Fast/Price-Sensitive': (0, 2, 0, 0),
        # etc...
    }
    
    heatmap_data = {}
    for archetype, state in archetypes.items():
        if state in q_agent.Q:
            heatmap_data[archetype] = q_agent.Q[state]
    
    return heatmap_data

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env = SalesEnvironment(max_steps=50)
    connection_id = id(websocket)
    environments[connection_id] = env
    episode_steps: list = []

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "reset":
                episode_steps.clear()
                result = env.reset()
                if hasattr(result, "model_dump"):
                    result = result.model_dump()
                elif hasattr(result, "__dict__") and not isinstance(result, dict):
                    result = vars(result)
                
                # Send agent stats with reset
                result["agent_stats"] = {
                    "epsilon": round(q_agent.epsilon, 3),
                    "states_learned": len(q_agent.Q),
                    "episodes": q_agent.episode_count,
                }
                
                await websocket.send_json({"type": "reset_response", **result})

            elif message_type == "step":
                action_data = data.get("action")
                
                # If no action provided, use Q-agent to choose one
                # if not action_data:
                #     obs = env._get_observation().model_dump()
                #     action_idx = q_agent.choose_action(obs)
                #     action_data = {"action_type": action_idx}
                
                chosen_action_idx = None
                if not action_data:
                    obs = env._get_observation().model_dump()
                    action_idx = q_agent.choose_action(obs)
                    chosen_action_idx = int(action_idx)          # <-- store it
                    action_data = {"action_type": action_idx}
                    
                try:
                    action = SalesAction(**action_data)
                    result = env.step(action)

                    if hasattr(result, "model_dump"):
                        result_dict = result.model_dump()
                    elif hasattr(result, "__dict__"):
                        result_dict = vars(result)
                    elif not isinstance(result, dict):
                        result_dict = dict(result)
                    else:
                        result_dict = result

                    # Q-learning update
                    reward = result_dict.get("reward", 0.0)
                    obs = result_dict.get("observation", {})
                    done = bool(result_dict.get("done", False))
                    
                    q_agent.update(reward, obs, done)
                    
                    log_takeaway(action, result_dict, episode_steps)

                    if done:
                        q_agent.end_episode()
                        q_agent.save(QTABLE_FILE)  # Save after each episode
                        await write_episode_summary(result_dict, episode_steps)
                        episode_steps.clear()

                    # Add agent stats to response
                    result_dict["agent_stats"] = {
                        "epsilon": round(q_agent.epsilon, 3),
                        "states_learned": len(q_agent.Q),
                        "learned_policy": q_agent.get_policy_summary() if done else None,
                    }
                    
                    result_dict["chosen_action"] = chosen_action_idx 

                    await websocket.send_json({"type": "step_response", **result_dict})

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif message_type == "get_action":
                # Frontend requests Q-agent's action choice
                obs = data.get("observation", {})
                action_idx = q_agent.choose_action(obs)
                await websocket.send_json({
                    "type": "action_response",
                    "action": action_idx
                })

            elif message_type == "state":
                state = env.state()
                await websocket.send_json({"type": "state_response", "state": state})

            elif message_type == "oversight_report":
                report = env.get_oversight_report()
                await websocket.send_json({
                    "type": "oversight_report_response",
                    "report": report.model_dump()
                })

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown: {message_type}"})

    except WebSocketDisconnect:
        if connection_id in environments:
            del environments[connection_id]
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
        finally:
            if connection_id in environments:
                del environments[connection_id]

@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    return r"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Persuasion RL with Fleet AI Oversight - Backend Q-Learning</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                   background: linear-gradient(135deg, #667eea 0%, black);
                   padding: 20px; min-height: 100vh; }
            .container { max-width: 1400px; margin: 0 auto; }
            h1 { color: white; text-align: center; margin-bottom: 10px; font-size: 2.5em; }
            .subtitle { color: rgba(255,255,255,0.9); text-align: center; margin-bottom: 30px; }
            .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 20px; }
            .panel { background: white; border-radius: 12px; padding: 25px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
            .panel h2 { color: #333; margin-bottom: 15px; border-bottom: 3px solid #667eea; padding-bottom: 10px; }
            .actions-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
            .action-btn {
                padding: 15px; border: 2px solid #ddd; border-radius: 8px;
                background: white; cursor: pointer; transition: all 0.3s;
                font-size: 14px; font-weight: 500; text-align: center;
            }
            .action-btn:hover { background: #f0f0f0; transform: translateY(-2px); }
            .action-btn.active {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; border-color: #667eea;
                transform: scale(1.05); box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
                animation: pulse 0.5s ease-in-out;
            }
            @keyframes pulse {
                0%, 100% { transform: scale(1.05); }
                50% { transform: scale(1.1); }
            }
            .metric {
                display: flex; justify-content: space-between; align-items: center;
                padding: 12px; margin: 8px 0; background: #f8f9fa; border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .metric-label { font-weight: 600; color: #555; }
            .metric-value { font-weight: bold; color: #667eea; font-size: 1.1em; }
            .engagement-score {
                font-size: 3em; text-align: center; color: #667eea;
                margin: 20px 0; font-weight: bold;
            }
            .engagement-label { text-align: center; color: #666; margin-bottom: 15px; }
            .controls { text-align: center; margin-top: 20px; }
            .btn {
                padding: 15px 30px; margin: 5px; border: none; border-radius: 8px;
                font-size: 16px; font-weight: 600; cursor: pointer;
                transition: all 0.3s; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3); }
            .btn-success { background: #28a745; color: white; }
            .btn-success:hover { background: #218838; }
            .btn-danger { background: #dc3545; color: white; }
            .btn-danger:hover { background: #c82333; }
            .btn:disabled { opacity: 0.5; cursor: not-allowed; }
            #status {
                padding: 15px; border-radius: 8px; text-align: center;
                margin-bottom: 20px; font-weight: 600;
            }
            .status-info    { background: #d1ecf1; color: #0c5460; }
            .status-success { background: #d4edda; color: #155724; }
            .status-warning { background: #fff3cd; color: #856404; }
            .status-danger  { background: #f8d7da; color: #721c24; }
            #feedback {
                background: #f8f9fa; padding: 20px; border-radius: 8px;
                margin-top: 15px; min-height: 100px; border-left: 4px solid #667eea;
            }
            .episode-result {
                text-align: center; padding: 30px; border-radius: 12px;
                margin: 20px 0; font-size: 1.3em;
            }
            .episode-won  { background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); }
            .episode-lost { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
            .log-panel {
                background: #f8f9fa; padding: 15px; border-radius: 8px;
                margin-top: 10px; max-height: 220px; overflow-y: auto;
                font-size: 12px; white-space: pre-wrap; font-family: monospace;
                color: #333; line-height: 1.5;
            }
            #oversightLogPanel { border-left: 4px solid #dc3545; }
            #takeawaysPanel    { border-left: 4px solid #764ba2; }
            #qtablePanel       { border-left: 4px solid #667eea; overflow-x: auto; white-space: pre; }
            .oversight-full-panel {
                background: white; border-radius: 12px; padding: 25px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2); margin-bottom: 20px;
            }
            .oversight-full-panel h2 {
                color: #333; margin-bottom: 15px;
                border-bottom: 3px solid #dc3545; padding-bottom: 10px;
            }
            .episode-badge {
                position: fixed; top: 20px; right: 20px;
                background: rgba(255,255,255,0.97); border-radius: 14px;
                padding: 14px 22px; box-shadow: 0 6px 30px rgba(0,0,0,0.22);
                z-index: 1000; text-align: center; min-width: 155px;
                border-top: 4px solid #667eea;
            }
            .episode-badge .ep-label  { font-size: 10px; font-weight: 700; color: #aaa; text-transform: uppercase; letter-spacing: 1.5px; }
            .episode-badge .ep-number { font-size: 2.6em; font-weight: bold; color: #667eea; line-height: 1.05; }
            .episode-badge .ep-step   { font-size: 11px; color: #bbb; margin-top: 2px; }
            .episode-badge .ep-wins   { font-size: 12px; font-weight: 700; margin-top: 4px; color: #28a745; }
            .episode-badge .ep-epsilon { font-size: 11px; color: #888; margin-top: 2px; }
        </style>
    </head>
    <body>
        <div class="container">

            <div class="episode-badge">
                <div class="ep-label">Episode</div>
                <div class="ep-number" id="badgeEpNum">0</div>
                <div class="ep-step"   id="badgeStep">Step — / 50</div>
                <div class="ep-wins"   id="badgeWins">0W / 0L</div>
                <div class="ep-epsilon" id="badgeEpsilon">ε = 0.500</div>
            </div>

            <h1>🎯 Persuasion RL Environment with Adversarial Hidden State</h1>
            <p class="subtitle">
                <strong>Goal:</strong> Achieve 10/10 Adversarial Trust Score by Inferring and Adapting to Adversarial Hidden State
            </p>

            <div id="status" class="status-info">Connecting...</div>

            <div class="grid">
                <div class="panel">
                    <h2>Adverserial Persuasion Agent Actions</h2>
                    <div class="actions-grid" id="actionsGrid"></div>
                    <div class="controls">
                        <button class="btn btn-primary" onclick="startAutoPlay()">&#9654; Auto-Play Episode</button>
                        <button class="btn btn-danger"  onclick="stopAutoPlay()">&#9646;&#9646; Stop</button>
                        <button class="btn btn-success" onclick="resetEnv()">&#128260; Reset</button>
                    </div>
                    <div id="feedback"></div>
                </div>

                <div class="panel">
                    <h2>Adversarial Trust Score</h2>
                    <div class="engagement-label">PRIMARY GOAL: Get to 10/10</div>
                    <div class="engagement-score" id="engagementScore">5.0/10</div>
                    <h2 style="margin-top: 30px;">Metrics</h2>
                    <div id="metrics"></div>
                    <h2 style="margin-top: 20px;">🕵️ Prospect Hidden State</h2>
                    <canvas id="prospectRadar" height="220"></canvas>
                    <h2 style="margin-top: 20px;">&#129504; Learned Policy (Backend)</h2>
                    <div class="log-panel" id="qtablePanel">Loading...</div>
                    <h2 style="margin-top: 20px;">&#128203; Episode Takeaways</h2>
                    <div class="log-panel" id="takeawaysPanel">No takeaways yet.</div>
                    <div style="text-align:center; margin-top: 8px;">
                        <button class="btn btn-primary" style="padding: 8px 18px; font-size: 13px;" onclick="fetchTakeaways()">&#128260; Refresh</button>
                    </div>
                </div>
            </div>

            <div style="display:grid; grid-template-columns: 3fr 2fr; gap: 20px; margin-bottom: 20px;">
                <div class="panel">
                    <h2>📈 Reward Curve</h2>
                    <p style="color:#666; font-size:12px; margin-bottom:10px;">
                        Raw reward per episode (bars) + 5-episode rolling average (line).
                        Backend Q-learning should show clear upward trend as it learns.
                    </p>
                    <canvas id="rewardChart" height="160"></canvas>
                </div>
                <div class="panel">
                    <h2>🎯 Learned Policy</h2>
                    <p style="color:#666; font-size:12px; margin-bottom:10px;">
                        Best action per stage learned by backend Q-agent.
                    </p>
                    <div id="optimalPolicy" style="font-family:monospace; font-size:13px; line-height:2;">
                        Loading from backend...
                    </div>
                </div>
            </div>

            <div class="oversight-full-panel">
                <h2>&#129302; Fleet AI Oversight — Cross-Episode Behavioural Analysis + Meta-Learning with LLM Code Generation</h2>
                <p style="color:#666; font-size:13px; margin-bottom:10px;">
                    Updated after every episode. Writes policy hints that seed new Q-table states.
                </p>
                <div class="log-panel" id="oversightLogPanel" style="max-height:320px;">
                    No oversight analysis yet. Complete an episode to see the Fleet AI report.
                </div>
                <div style="text-align:center; margin-top: 10px;">
                    <button class="btn btn-primary" style="padding: 8px 18px; font-size: 13px;" onclick="fetchOversight()">&#128260; Refresh Oversight Report</button>
                </div>
            </div>

            <div id="episodeResult"></div>
        </div>

        <script>
            let ws;
            let connected   = false;
            let autoPlaying = false;
            let currentStep = 0;

            const actionNames = [
                "Research Prospect","Send Intro Email","Send Follow-Up","Schedule Call",
                "Send Proposal","Customize Proposal","Negotiate Price","Send Contract",
                "Update CRM","Do Nothing"
            ];

            let episodeCount = 0;
            let wins = 0;
            let losses = 0;
            let currentEpsilon = 0.5;

            let rewardHistory = [];
            let rewardChart   = null;

            let prospectRadarChart = null;

            function initProspectRadar() {
                const ctx = document.getElementById('prospectRadar').getContext('2d');
                prospectRadarChart = new Chart(ctx, {
                    type: 'radar',
                    data: {
                        labels: ['Price Sensitivity', 'Relationship Need', 'Decision Speed', 'Budget', 'Warmth'],
                        datasets: [{
                            label: 'Prospect Profile',
                            data: [0, 0, 0, 0, 0],
                            backgroundColor: 'rgba(220, 53, 69, 0.2)',
                            borderColor: 'rgba(220, 53, 69, 1)',
                            borderWidth: 2,
                            pointBackgroundColor: 'rgba(220, 53, 69, 1)',
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            r: {
                                min: 0, max: 100,
                                ticks: { stepSize: 25, font: { size: 9 } },
                                pointLabels: { font: { size: 10 } }
                            }
                        },
                        plugins: { legend: { display: false } }
                    }
                });
            }

            function updateProspectRadar(obs) {
                const p = obs.prospect;
                if (!p || !prospectRadarChart) return;

                const speedMap = { 'fast': 90, 'medium': 55, 'slow': 20 };

                const ps        = (p.price_sensitivity   ?? 0.5) * 100;       // 0–1 → 0–100
                const ri        = (p.relationship_importance ?? 0.5) * 100;   // 0–1 → 0–100
                const speed     = speedMap[p.decision_speed] ?? 50;            // string → score
                const budget    = Math.min(100, (p.budget ?? 1000) / 200);    // normalise; adjust 200 to your max budget
                const warmth    = obs.prospect_warmth ?? 50;                   // already 0–100

                prospectRadarChart.data.datasets[0].data = [ps, ri, speed, budget, warmth];
                prospectRadarChart.update('none');
            }
            

            function createActionButtons() {
                const grid = document.getElementById('actionsGrid');
                grid.innerHTML = '';
                actionNames.forEach((name, idx) => {
                    const btn = document.createElement('div');
                    btn.className = 'action-btn';
                    btn.id = 'action-' + idx;
                    btn.textContent = name;
                    grid.appendChild(btn);
                });
            }

            function connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(protocol + '//' + window.location.host + '/ws');
                ws.onopen    = () => { connected = true; showStatus('Connected! Backend Q-agent ready. Click Auto-Play.', 'success'); };
                ws.onmessage = (event) => { handleResponse(JSON.parse(event.data)); };
                ws.onerror   = () => { showStatus('Connection error. Retrying...', 'danger'); setTimeout(connect, 2000); };
                ws.onclose   = () => { connected = false; showStatus('Disconnected. Reconnecting...', 'warning'); setTimeout(connect, 2000); };
            }

            


            function resetEnv() {
                if (!connected || !ws || ws.readyState !== WebSocket.OPEN) return;
                stopAutoPlay();
                ws.send(JSON.stringify({ type: 'reset' }));
                document.getElementById('episodeResult').innerHTML = '';
                showStatus('Resetting environment...', 'info');
            }

            function highlightAction(actionType) {
                document.querySelectorAll('.action-btn').forEach(b => b.classList.remove('active'));
                const btn = document.getElementById('action-' + actionType);
                if (btn) { btn.classList.add('active'); setTimeout(() => btn.classList.remove('active'), 300); }
            }

            function startAutoPlay() {
                if (!connected) { showStatus('Not connected!', 'danger'); return; }
                autoPlaying = true;
                currentStep = 0;
                document.getElementById('episodeResult').innerHTML = '';
                showStatus('Agent starting episode...', 'info');
                ws.send(JSON.stringify({ type: 'reset' }));
            }

            

            function stopAutoPlay() {
                autoPlaying = false;
                showStatus('Auto-play stopped', 'warning');
            }

            function initRewardChart() {
                const ctx = document.getElementById('rewardChart').getContext('2d');
                rewardChart = new Chart(ctx, {
                    data: {
                        labels: [],
                        datasets: [
                            {
                                type: 'bar',
                                label: 'Episode Reward',
                                data: [],
                                backgroundColor: [],
                                borderColor: [],
                                borderWidth: 1,
                                yAxisID: 'y',
                            },
                            {
                                type: 'line',
                                label: '5-ep Rolling Avg',
                                data: [],
                                borderColor: '#28a745',
                                backgroundColor: 'rgba(40,167,69,0.08)',
                                borderWidth: 2.5,
                                pointRadius: 3,
                                tension: 0.4,
                                fill: true,
                                yAxisID: 'y',
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        interaction: { mode: 'index', intersect: false },
                        plugins: {
                            legend: { position: 'top', labels: { font: { size: 11 } } },
                            tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) } }
                        },
                        scales: {
                            x: { title: { display: true, text: 'Episode', font: { size: 11 } } },
                            y: { title: { display: true, text: 'Reward',  font: { size: 11 } } }
                        }
                    }
                });
            }

            function rollingAvg(arr, window) {
                return arr.map((_, i) => {
                    const slice = arr.slice(Math.max(0, i - window + 1), i + 1);
                    return slice.reduce((a, b) => a + b, 0) / slice.length;
                });
            }

            function updateRewardChart(reward) {
                rewardHistory.push(reward);
                if (!rewardChart) initRewardChart();
                const labels  = rewardHistory.map((_, i) => 'Ep ' + (i + 1));
                const rolling = rollingAvg(rewardHistory, 5);
                const colors  = rewardHistory.map(r => r >= 0 ? 'rgba(102,126,234,0.55)' : 'rgba(220,53,69,0.55)');
                const borders = rewardHistory.map(r => r >= 0 ? 'rgba(102,126,234,1)'    : 'rgba(220,53,69,1)');
                rewardChart.data.labels                      = labels;
                rewardChart.data.datasets[0].data            = rewardHistory;
                rewardChart.data.datasets[0].backgroundColor = colors;
                rewardChart.data.datasets[0].borderColor     = borders;
                rewardChart.data.datasets[1].data            = rolling;
                rewardChart.update('none');
            }

            function updateOptimalPolicy(learned_policy) {
                if (!learned_policy) return;
                
                const stageEmojis = {'LEAD': '🔍', 'QUALIFIED': '✅', 'PROPOSAL': '📄', 'NEGOTIATION': '🤝'};
                let html = '';
                
                for (const [stage, info] of Object.entries(learned_policy)) {
                    const emoji = stageEmojis[stage] || '';
                    const qVal = info.q_value;
                    const color = qVal > 5 ? '#28a745' : qVal > 0 ? '#ffc107' : '#dc3545';
                    
                    html += '<div style="display:flex;justify-content:space-between;align-items:center;' +
                            'padding:7px 10px;margin:3px 0;background:#f8f9fa;border-radius:6px;' +
                            'border-left:3px solid ' + color + ';">' +
                            '<span style="color:#555;font-size:11px;">' + emoji + ' ' + stage + '</span>' +
                            '<span style="font-weight:700;color:' + color + ';font-size:12px;">' + info.action + '</span>' +
                            '<span style="color:#aaa;font-size:10px;">Q=' + qVal.toFixed(1) + '</span>' +
                            '</div>';
                }
                
                document.getElementById('optimalPolicy').innerHTML = html;
            }

            function updateBadge(step, stats) {
                document.getElementById('badgeEpNum').textContent = episodeCount;
                document.getElementById('badgeStep').textContent  = 'Step ' + step + ' / 50';
                document.getElementById('badgeWins').textContent  = wins + 'W / ' + losses + 'L';
                if (stats && stats.epsilon !== undefined) {
                    document.getElementById('badgeEpsilon').textContent = 'ε = ' + stats.epsilon.toFixed(3);
                }
            }

            function handleResponse(data) {
                if (data.type === 'reset_response') {
                    const obs = data.observation;
                    const stats = data.agent_stats;
                    
                    if (obs) updateMetrics(obs);
                    if (stats) {
                        updateAgentPanel(stats);
                        currentEpsilon = stats.epsilon;
                    }
                    
                    if (autoPlaying) {
                        currentStep = 0;
                        // Backend will choose action - just send step request with no action
                        setTimeout(() => ws.send(JSON.stringify({ type: 'step' })), 350);
                    }
                    return;
                }

                if (data.type === 'step_response') {

                    if (data.chosen_action !== null && data.chosen_action !== undefined) {
                        highlightAction(data.chosen_action);
                    }
                    const obs = data.observation;
                    const stats = data.agent_stats;
                    
                    if (obs) {
                        updateMetrics(obs);
                        updateProspectRadar(obs);   // ADD THIS

                        updateFeedback(obs, data.reward);
                    }
                    
                    if (stats) {
                        updateAgentPanel(stats);
                        currentEpsilon = stats.epsilon;
                    }

                    if (data.done) {
                        const _epWon = obs && obs.deal_stage === 4;
                        const _epReward = data.state && data.state.total_reward != null ? data.state.total_reward : 0;
                        
                        episodeCount++;
                        if (_epWon) wins++; else losses++;
                        
                        const wasAutoPlaying = autoPlaying;
                        autoPlaying = false;
                        
                        updateRewardChart(_epReward);
                        if (stats && stats.learned_policy) {
                            updateOptimalPolicy(stats.learned_policy);
                        }
                        
                        showEpisodeResult(obs, data.state, wasAutoPlaying);
                        setTimeout(() => { fetchTakeaways(); fetchOversight(); }, 3500);
                        return;
                    }

                    currentStep++;
                    if (autoPlaying) {
                        // Backend chooses action
                        setTimeout(() => ws.send(JSON.stringify({ type: 'step' })), 350);
                    }
                    return;
                }

                if (data.type === 'error') {
                    console.error(data.message);
                    showStatus('Error: ' + data.message, 'danger');
                    autoPlaying = false;
                }
            }

            function updateAgentPanel(stats) {
                const lines = [
                    `Backend Q-Learning Agent`,
                    `States learned: ${stats.states_learned}`,
                    `Episodes trained: ${stats.episodes}`,
                    `Epsilon (exploration): ${stats.epsilon.toFixed(3)}`,
                    ``,
                    `Agent chooses actions via epsilon-greedy`,
                    `policy using learned Q-values.`
                ];
                document.getElementById('qtablePanel').textContent = lines.join('\n');
            }

            function updateMetrics(obs) {
                const engScore = document.getElementById('engagementScore');
                engScore.textContent = obs.engagement_score.toFixed(1) + '/10';
                engScore.style.color = obs.engagement_score >= 8 ? '#28a745' : obs.engagement_score >= 6 ? '#ffc107' : '#dc3545';
                const stages = ['LEAD','QUALIFIED','PROPOSAL','NEGOTIATION','CLOSED_WON','CLOSED_LOST'];
                document.getElementById('metrics').innerHTML =
                    '<div class="metric"><span class="metric-label">Deal Stage:</span><span class="metric-value">' + stages[obs.deal_stage] + '</span></div>' +
                    '<div class="metric"><span class="metric-label">Prospect Warmth:</span><span class="metric-value">' + obs.prospect_warmth.toFixed(0) + '/100</span></div>' +
                    '<div class="metric"><span class="metric-label">Step:</span><span class="metric-value">' + obs.episode_step + '/50</span></div>';
                updateBadge(obs.episode_step, {epsilon: currentEpsilon});
            }

            function updateFeedback(obs, reward) {
                document.getElementById('feedback').innerHTML =
                    '<strong>Last Action Feedback:</strong><br>' + obs.action_feedback + '<br><br>' +
                    '<strong>Reward:</strong> ' + (reward > 0 ? '+' : '') + reward.toFixed(2) + '<br>' +
                    '<strong>Engagement:</strong> ' + obs.engagement_score.toFixed(1) + '/10';
            }

            function showEpisodeResult(obs, state, wasAutoPlaying) {
                const isWon = obs.deal_stage === 4;
                const resultDiv = document.getElementById('episodeResult');
                resultDiv.className = 'panel episode-result ' + (isWon ? 'episode-won' : 'episode-lost');
                resultDiv.innerHTML =
                    '<h2>' + (isWon ? 'DEAL WON!' : 'Deal Lost') + '</h2>' +
                    '<p style="margin-top:15px;"><strong>Final Trust Score:</strong> ' + obs.engagement_score.toFixed(1) + '/10<br>' +
                    '<strong>Total Reward:</strong> ' + state.total_reward.toFixed(2) + '<br>' +
                    '<strong>Steps Taken:</strong> ' + state.step_count + '/50</p>' +
                    (obs.engagement_score >= 8 ? '<p style="margin-top:15px;font-size:1.2em;">EXCELLENT ENGAGEMENT!</p>' : '') +
                    '<p style="margin-top:20px;font-style:italic;color:#555;">Backend Q-agent learned from this episode. Fleet AI Oversight updating...' +
                    (wasAutoPlaying ? ' Next episode in 4 seconds.' : '') + '</p>';
                showStatus('Episode complete! ' + (isWon ? 'Won' : 'Lost') + ' | Engagement: ' + obs.engagement_score.toFixed(1) + '/10 | Epsilon: ' + currentEpsilon.toFixed(3), isWon ? 'success' : 'warning');
                
                // Highlight the action that was chosen (if we knew it)
                // Since backend chooses, we don't explicitly show it here
                
                if (wasAutoPlaying) setTimeout(() => startAutoPlay(), 4000);
            }

            async function fetchTakeaways() {
                try {
                    const data  = await fetch('/takeaways').then(r => r.json());
                    const panel = document.getElementById('takeawaysPanel');
                    panel.textContent = data.content || 'No takeaways yet.';
                    panel.scrollTop   = panel.scrollHeight;
                } catch (e) { console.error('Failed to fetch takeaways:', e); }
            }

            async function fetchOversight() {
                try {
                    const data  = await fetch('/oversight').then(r => r.json());
                    const panel = document.getElementById('oversightLogPanel');
                    panel.textContent = data.content || 'No oversight analysis yet.';
                    panel.scrollTop   = panel.scrollHeight;
                } catch (e) { console.error('Failed to fetch oversight:', e); }
            }

            function showStatus(message, type) {
                const status = document.getElementById('status');
                status.textContent = message;
                status.className   = 'status-' + type;
            }

            document.addEventListener('DOMContentLoaded', () => {
                createActionButtons();
                initProspectRadar();
                connect();
                updateBadge(0, {epsilon: 0.5});
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)