"""
Pydantic models for Sales RL Environment.
"""

from pydantic import BaseModel
from typing import Optional, List
from enum import IntEnum


class ActionType(IntEnum):
    """Sales actions available to the agent."""
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


class DealStage(IntEnum):
    """B2B deal pipeline stages."""
    LEAD = 0
    QUALIFIED = 1
    PROPOSAL = 2
    NEGOTIATION = 3
    CLOSED_WON = 4
    CLOSED_LOST = 5


class ProspectInfo(BaseModel):
    """Prospect personality and company details."""
    company_size: str  # startup, small, medium, enterprise
    industry: str
    budget: float
    decision_speed: str  # slow, medium, fast
    price_sensitivity: float  # 0.0 - 1.0
    relationship_importance: float  # 0.0 - 1.0


class SalesAction(BaseModel):
    """Action taken by the agent."""
    action_type: ActionType


class SalesObservation(BaseModel):
    """Observation returned after each step."""
    engagement_score: float
    deal_stage: DealStage
    prospect_warmth: float
    days_since_contact: int
    emails_sent: int
    calls_completed: int
    proposal_sent: bool
    discount_offered: float
    contract_sent: bool
    prospect: ProspectInfo
    action_feedback: str
    win_probability: Optional[float] = None
    episode_step: int
    episode_done: bool
    oversight_alert: Optional[str] = None
    behavior_score: float


class SalesState(BaseModel):
    """Complete state of the environment."""
    episode_id: str
    step_count: int
    total_reward: float
    engagement_score: float
    deal_outcome: Optional[str] = None
    action_sequence: List[ActionType]


class OversightReport(BaseModel):
    """Fleet AI oversight analysis."""
    episode_id: str
    issues_detected: List[str]
    recommendations: List[str]
    behavior_score: float
    suspicious_patterns: List[str]