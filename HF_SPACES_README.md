---
title: Sales RL Environment with Fleet AI Oversight
emoji: 🎯
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
tags:
  - reinforcement-learning
  - openenv
  - sales
  - oversight
  - fleet-ai
---

# Sales RL Environment with Fleet AI Oversight

**OpenEnv 0.2.1 Compatible | Scale AI Hackathon ($10K Prize)**

## 🎯 Goal: Achieve 10/10 Engagement Score

This is a B2B Sales RL environment with a Fleet AI oversight agent that monitors the sales agent's behavior for ethics and best practices.

### Features

- **Auto-playing visual interface** with action highlighting
- **Fleet AI oversight** monitoring for spam, premature discounting, poor timing
- **10/10 engagement score** as primary optimization goal
- **Learning loop** - agent learns from each episode
- **OpenEnv 0.2.1 compatible** - ready for TRL/Unsloth training

### Quick Start

1. **Open the web interface**: Click "App" above or visit `/web`
2. **Click "▶ Auto-Play Episode"** to watch the agent learn
3. **Watch actions highlight** as they're executed (1-2 sec each)
4. **See Fleet AI alerts** when oversight detects issues
5. **Episode ends** when deal closes (won/lost)
6. **Agent learns** and resets for next episode

### For Training

Connect to this Space via WebSocket for RL training:

```python
from websockets import connect

ENV_URL = "wss://your-username-sales-env.hf.space/ws"

async with connect(ENV_URL) as ws:
    # Send reset
    await ws.send(json.dumps({"type": "reset"}))
    
    # Send action
    await ws.send(json.dumps({
        "type": "step",
        "action": {"action_type": 1}  # 0-9
    }))
```

See full training script with Unsloth + TRL in the GitHub repo.

### Scale AI Subtheme

**Perfect alignment** with "Long Horizon Business Workflows (Sales)":
- ✅ Sales focus
- ✅ Long horizon (10-50 steps)
- ✅ Business workflows
- ✅ Non-code operations

### Tech Stack

- **OpenEnv 0.2.1** - Environment framework
- **FastAPI + WebSockets** - Real-time communication
- **Fleet AI** - Oversight agent
- **Pydantic** - Type-safe models
- **Docker** - HF Spaces deployment

Built for the OpenEnv Hackathon (March 2026)
