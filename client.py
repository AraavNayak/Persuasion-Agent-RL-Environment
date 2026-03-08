"""
Sales Environment Client for OpenEnv 0.2.1
"""

import asyncio
import json
from typing import Optional, Dict, Any
import websockets
from models import SalesAction

class SalesEnv:
    """Async client for Sales RL Environment"""
    
    def __init__(self, base_url: str = "ws://localhost:8000"):
        if base_url.startswith("http://"):
            self.ws_url = base_url.replace("http://", "ws://") + "/ws"
        elif base_url.startswith("https://"):
            self.ws_url = base_url.replace("https://", "wss://") + "/ws"
        else:
            self.ws_url = base_url + "/ws" if not base_url.endswith("/ws") else base_url
        
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
    
    async def __aenter__(self):
        self.websocket = await websockets.connect(self.ws_url)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.websocket:
            await self.websocket.close()
    
    async def reset(self) -> Dict[str, Any]:
        """Reset environment"""
        if not self.websocket:
            raise RuntimeError("Not connected")
        
        await self.websocket.send(json.dumps({"type": "reset"}))
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if data.get("type") == "error":
            raise RuntimeError(f"Reset failed: {data.get('message')}")
        
        return {
            "observation": data["observation"],
            "state": data["state"],
            "reward": data["reward"],
            "done": data["done"]
        }
    
    async def step(self, action: SalesAction) -> Dict[str, Any]:
        """Take a step"""
        if not self.websocket:
            raise RuntimeError("Not connected")
        
        await self.websocket.send(json.dumps({
            "type": "step",
            "action": action.model_dump()
        }))
        
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if data.get("type") == "error":
            raise RuntimeError(f"Step failed: {data.get('message')}")
        
        return {
            "observation": data["observation"],
            "state": data["state"],
            "reward": data["reward"],
            "done": data["done"]
        }
    
    async def get_oversight_report(self) -> Dict[str, Any]:
        """Get Fleet AI oversight report"""
        if not self.websocket:
            raise RuntimeError("Not connected")
        
        await self.websocket.send(json.dumps({"type": "oversight_report"}))
        response = await self.websocket.recv()
        data = json.loads(response)
        
        return data.get("report")
    
    def sync(self):
        """Get synchronous wrapper"""
        return SyncSalesEnv(self)

class SyncSalesEnv:
    """Synchronous wrapper"""
    
    def __init__(self, async_client: SalesEnv):
        self.async_client = async_client
        self.loop = None
    
    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.async_client.__aenter__())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loop.run_until_complete(self.async_client.__aexit__(exc_type, exc_val, exc_tb))
        self.loop.close()
    
    def reset(self) -> Dict[str, Any]:
        return self.loop.run_until_complete(self.async_client.reset())
    
    def step(self, action: SalesAction) -> Dict[str, Any]:
        return self.loop.run_until_complete(self.async_client.step(action))
    
    def get_oversight_report(self) -> Dict[str, Any]:
        return self.loop.run_until_complete(self.async_client.get_oversight_report())
