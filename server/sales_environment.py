"""
Sales Environment with Fleet AI Oversight Agent.
OpenEnv 0.2.1 compatible.

Key improvements over v1:
- Prospect personality actually affects rewards and transitions
- Reward function is sparse and outcome-driven, not hardcoded constants
- Learning loop closed: oversight agent writes action_weights that seed
  the next episode's policy hints, returned in reset() observation
- Stage transitions are probabilistic based on prospect traits
- Dense shaping removed; agent must discover what works per prospect type
"""

import uuid
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from models import (
    SalesAction, SalesObservation, SalesState, ProspectInfo, OversightReport,
    DealStage, ActionType
)

# Persisted policy hints written by the oversight agent between episodes
POLICY_HINTS_FILE = Path(__file__).parent / "policy_hints.json"


def _load_policy_hints() -> Dict:
    """Load oversight agent's cross-episode policy hints if they exist."""
    if POLICY_HINTS_FILE.exists():
        try:
            return json.loads(POLICY_HINTS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_policy_hints(hints: Dict):
    """Called by app.py oversight agent to persist hints for next episode."""
    try:
        POLICY_HINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        POLICY_HINTS_FILE.write_text(json.dumps(hints, indent=2))
    except Exception as e:
        print(f"[policy_hints] write failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Prospect personality profiles — drive reward multipliers & transition probs
# ---------------------------------------------------------------------------

PERSONALITY_PROFILES = {
    # (decision_speed, price_sensitivity bucket) -> trait dict
    # These create genuinely different optimal strategies per prospect
}

def _prospect_reward_multipliers(prospect: ProspectInfo) -> Dict[str, float]:
    """
    Convert prospect personality into per-action reward multipliers.
    This is what makes the reward function non-trivial — the same action
    is worth different amounts depending on who you're selling to.
    """
    ps = prospect.price_sensitivity        # 0=not sensitive, 1=very sensitive
    ri = prospect.relationship_importance  # 0=transactional, 1=relationship-driven
    ds = prospect.decision_speed           # 'slow', 'medium', 'fast'

    speed_factor = {'slow': 0.7, 'medium': 1.0, 'fast': 1.4}.get(ds, 1.0)

    return {
        # Relationship-driven prospects reward calls & customization much more
        ActionType.SCHEDULE_CALL:       1.0 + ri * 1.5,
        ActionType.CUSTOMIZE_PROPOSAL:  1.0 + ri * 2.0,
        ActionType.SEND_INTRO_EMAIL:    1.0 + ri * 0.5,
        ActionType.SEND_FOLLOW_UP:      1.0 + ri * 0.3,

        # Price-sensitive prospects punish discounting less but reward research more
        ActionType.RESEARCH_PROSPECT:   1.0 + ps * 1.0,
        ActionType.NEGOTIATE_PRICE:     1.0 - ps * 0.4,  # high PS = discounts expected, less impressive

        # Fast prospects reward speed — proposals and contracts matter more
        ActionType.SEND_PROPOSAL:       speed_factor,
        ActionType.SEND_CONTRACT:       speed_factor,

        # CRM is always low-value but consistent
        ActionType.UPDATE_CRM:          1.0,
        ActionType.DO_NOTHING:          1.0,
    }


def _prospect_transition_probs(prospect: ProspectInfo) -> Dict[str, float]:
    """
    Prospect traits affect how hard stage transitions are.
    Enterprise/slow prospects need more warming before they'll advance.
    """
    ds = prospect.decision_speed
    size = prospect.company_size
    ri = prospect.relationship_importance

    # Warmth threshold to advance from LEAD -> QUALIFIED
    lead_threshold = {
        'startup': 45, 'small': 50, 'medium': 60, 'enterprise': 70
    }.get(size, 55)

    # Calls needed before QUALIFIED -> PROPOSAL
    calls_needed = {'slow': 2, 'medium': 1, 'fast': 1}.get(ds, 1)

    # Engagement threshold to enter NEGOTIATION
    # Relationship-driven prospects need higher engagement before negotiating
    negotiation_engagement = 6.0 + ri * 2.0  # 6.0 – 8.0

    return {
        'lead_warmth_threshold':      lead_threshold,
        'calls_needed':               calls_needed,
        'negotiation_engagement':     negotiation_engagement,
    }


# ---------------------------------------------------------------------------
# Oversight agent (rule-based, in-process)
# ---------------------------------------------------------------------------

class FleetAIOversightAgent:
    """
    In-process oversight that monitors behavior and writes structured
    policy hints back to disk so the next episode starts informed.

    The LLM-powered cross-episode analysis in app.py reads episode_history
    and calls save_policy_hints() with a JSON blob that reset() picks up.
    This is the closed loop: LLM oversight -> policy_hints.json -> reset().
    """

    def __init__(self):
        self.alerts = []
        self.action_history = []

    def analyze_action(self, action: ActionType, context: Dict) -> Tuple[Optional[str], float, List[str]]:
        issues = []
        alert = None
        score = 100.0

        self.action_history.append({
            'action': action,
            'stage':  context.get('deal_stage'),
            'warmth': context.get('prospect_warmth'),
            'step':   context.get('step'),
        })

        # Email spam
        if action == ActionType.SEND_FOLLOW_UP:
            recent = sum(1 for h in self.action_history[-5:]
                         if h['action'] in [ActionType.SEND_INTRO_EMAIL, ActionType.SEND_FOLLOW_UP])
            if recent >= 3:
                issues.append("Email spam — too many emails in short period")
                alert = "⚠️ Oversight: Slow down on emails. Prospect may disengage."
                score -= 30

        # Premature discounting
        if action == ActionType.NEGOTIATE_PRICE:
            if context.get('deal_stage', 0) < DealStage.NEGOTIATION:
                issues.append("Premature discounting before establishing value")
                alert = "⚠️ Oversight: Discounting too early. Build value first."
                score -= 25

        # Contract timing
        if action == ActionType.SEND_CONTRACT:
            if context.get('deal_stage', 0) != DealStage.NEGOTIATION:
                issues.append("Contract sent at wrong stage")
                alert = "⚠️ Oversight: Contract timing is off."
                score -= 20

        # Inaction on warm prospect
        if action == ActionType.DO_NOTHING:
            if context.get('prospect_warmth', 0) > 70:
                issues.append("Inaction when prospect is highly engaged")
                alert = "⚠️ Oversight: Prospect is warm! Take action now."
                score -= 15

        # Excessive discounting
        if context.get('discount_offered', 0) > 25:
            issues.append("Excessive discounting hurts margins")
            alert = "⚠️ Oversight: Discount too high. Protect margins."
            score -= 10

        # Good patterns
        if action == ActionType.RESEARCH_PROSPECT and context.get('step', 99) < 3:
            score += 5
        if action == ActionType.SCHEDULE_CALL and context.get('prospect_warmth', 0) > 60:
            score += 10

        return alert, max(0.0, min(100.0, score)), issues

    def generate_episode_report(self, episode_id: str, final_outcome: str) -> OversightReport:
        action_counts: Dict = {}
        for entry in self.action_history:
            a = entry['action']
            action_counts[a] = action_counts.get(a, 0) + 1

        suspicious_patterns = []
        recommendations = []

        if any(c > 5 for c in action_counts.values()):
            suspicious_patterns.append("Repetitive action pattern detected")
            recommendations.append("Diversify sales tactics")

        stages_seen = set(e['stage'] for e in self.action_history)
        if DealStage.QUALIFIED not in stages_seen and len(self.action_history) > 5:
            suspicious_patterns.append("Skipped qualification stage")
            recommendations.append("Always qualify leads before proposing")

        avg_score = 75.0
        if final_outcome == "won":
            avg_score += 15
        elif final_outcome == "lost":
            avg_score -= 10
        if suspicious_patterns:
            avg_score -= 20

        return OversightReport(
            episode_id=episode_id,
            issues_detected=[],
            recommendations=recommendations or ["Good job! Keep it up."],
            behavior_score=max(0.0, min(100.0, avg_score)),
            suspicious_patterns=suspicious_patterns,
        )

    def reset(self):
        self.action_history = []
        self.alerts = []


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class SalesEnvironment:
    """
    B2B Sales RL Environment with prospect-aware rewards and closed learning loop.

    What's different from v1:
    ─────────────────────────
    1. _execute_action() applies prospect-specific multipliers — the same
       action yields different rewards depending on who you're selling to.
       The agent must learn different strategies per prospect archetype.

    2. _check_stage_transition() uses prospect-derived thresholds — enterprise
       slow-movers need more calls and warmth before advancing.

    3. reset() returns policy_hints from the oversight agent's last write.
       The JS Q-table can be seeded from these hints (or an LLM-as-policy
       agent can receive them as a system prompt injection).

    4. Reward shaping is sparse: intermediate rewards are small signals,
       the big payoff is at the end based on actual outcome + engagement.
       This forces the agent to optimize for outcomes, not action counts.
    """

    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
        self.oversight_agent = FleetAIOversightAgent()
        self._reset_state()

    def _reset_state(self):
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.total_reward = 0.0
        self.action_sequence = []

        self.deal_stage = DealStage.LEAD
        self.prospect_warmth = 50.0
        self.engagement_score = 5.0
        self.days_since_contact = 0
        self.emails_sent = 0
        self.calls_completed = 0
        self.proposal_sent = False
        self.discount_offered = 0.0
        self.contract_sent = False

        self.prospect = self._generate_prospect()
        self._negotiation_entry_step = None

        # Pre-compute prospect-specific reward multipliers and transition thresholds
        self._reward_mults = _prospect_reward_multipliers(self.prospect)
        self._trans_probs  = _prospect_transition_probs(self.prospect)

        self.oversight_agent.reset()
        self.last_action_feedback = "Episode started. New prospect awaits."
        self.current_oversight_alert = None
        self.behavior_score = 100.0

        # Load any policy hints the LLM oversight agent wrote last episode
        self._policy_hints = _load_policy_hints()

    def _generate_prospect(self) -> ProspectInfo:
        company_sizes = ['startup', 'small', 'medium', 'enterprise']
        industries    = ['tech', 'finance', 'healthcare', 'retail', 'manufacturing']
        size          = np.random.choice(company_sizes)
        industry      = np.random.choice(industries)

        budget_ranges = {
            'startup':    (10,  50),
            'small':      (50,  200),
            'medium':     (200, 500),
            'enterprise': (500, 2000),
        }
        budget = float(np.random.uniform(*budget_ranges[size]))

        return ProspectInfo(
            company_size=size,
            industry=industry,
            budget=budget,
            decision_speed=np.random.choice(['slow', 'medium', 'fast']),
            price_sensitivity=float(np.random.uniform(0.3, 0.9)),
            relationship_importance=float(np.random.uniform(0.4, 1.0)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Dict:
        self._reset_state()
        obs   = self._get_observation()
        state = self._get_state()
        return {
            "observation":   obs.model_dump(),
            "state":         state.model_dump(),
            "reward":        0.0,
            "done":          False,
            # Policy hints from LLM oversight — JS can use these to seed Q-table
            "policy_hints":  self._policy_hints,
        }

    def step(self, action: SalesAction) -> Dict:
        self.step_count += 1
        self.days_since_contact += 1
        self.action_sequence.append(action.action_type)

        if self.step_count >= self.max_steps:
            obs   = self._get_observation(done=True)
            state = self._get_state()
            return {
                "observation": obs.model_dump(),
                "state":       state.model_dump(),
                "reward":      -10.0,
                "done":        True,
                "policy_hints": self._policy_hints,
            }

        # Oversight analysis before action
        context = {
            'deal_stage':       self.deal_stage,
            'prospect_warmth':  self.prospect_warmth,
            'discount_offered': self.discount_offered,
            'step':             self.step_count,
        }
        oversight_alert, behavior_score, _ = self.oversight_agent.analyze_action(
            action.action_type, context
        )
        self.current_oversight_alert = oversight_alert
        self.behavior_score = behavior_score

        # Execute action with prospect-aware rewards
        reward, feedback = self._execute_action(action)
        self.last_action_feedback = feedback

        # Oversight penalty (scaled, not flat)
        if oversight_alert:
            reward -= 3 + (100 - behavior_score) * 0.05

        self.total_reward += reward
        self._update_engagement()
        self._check_stage_transition()

        done = self.deal_stage in [DealStage.CLOSED_WON, DealStage.CLOSED_LOST]

        terminal_reward = 0.0
        if done:
            terminal_reward = self._compute_terminal_reward()
            self.total_reward += terminal_reward
            reward += terminal_reward

        # Lighter time penalty — only if agent is stuck early
        if self.step_count % 10 == 0 and self.deal_stage <= DealStage.LEAD:
            reward -= 3

        obs   = self._get_observation(done=done)
        state = self._get_state()

        if (self.deal_stage == DealStage.NEGOTIATION and not self.contract_sent
                and self._negotiation_entry_step is not None):
            steps_in_negotiation = self.step_count - self._negotiation_entry_step
            if steps_in_negotiation > 5:
                reward -= (steps_in_negotiation - 5) * 0.5
                
        return {
            "observation":  obs.model_dump(),
            "state":        state.model_dump(),
            "reward":       float(reward),
            "done":         done,
            "policy_hints": self._policy_hints,
        }

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _execute_action(self, action: SalesAction) -> Tuple[float, str]:
        """
        Prospect-aware reward execution.

        Base rewards are intentionally small — the big signal is terminal.
        Multipliers from _prospect_reward_multipliers() make the same action
        worth different amounts depending on prospect personality.
        """
        at   = action.action_type
        mult = self._reward_mults.get(at, 1.0)
        ps   = self.prospect.price_sensitivity
        ri   = self.prospect.relationship_importance

        reward   = 0.0
        feedback = ""

        if at == ActionType.RESEARCH_PROSPECT:
            # Always useful, more so for price-sensitive prospects
            self.prospect_warmth += 3 + ps * 4
            reward   = (2 + ps * 2) * mult
            feedback = f"Researched prospect. {'Price point noted.' if ps > 0.6 else 'Relationship angle identified.'}"

        elif at == ActionType.SEND_INTRO_EMAIL:
            if self.emails_sent == 0:
                self.emails_sent += 1
                self.days_since_contact = 0
                warmth_gain = 8 + ri * 6  # relationship prospects respond more to personal outreach
                self.prospect_warmth += warmth_gain
                reward   = 5 * mult
                feedback = f"Intro sent. {'Warm response — they value relationships.' if ri > 0.7 else 'Professional acknowledgement received.'}"
            else:
                reward   = -2
                feedback = "Already introduced. Second intro looks disorganized."

        elif at == ActionType.SEND_FOLLOW_UP:
            if self.emails_sent > 0 and self.days_since_contact > 3:
                self.emails_sent += 1
                self.days_since_contact = 0
                self.prospect_warmth += 3 + ri * 3
                reward   = 3 * mult
                feedback = "Follow-up landed. Keeping momentum."
            elif self.days_since_contact <= 3:
                reward   = -4
                feedback = "Too soon. They haven't had time to think."
            else:
                reward   = -2
                feedback = "No intro sent yet. Follow-up before intro confuses them."

        elif at == ActionType.SCHEDULE_CALL:
            tp = self._trans_probs
            if self.deal_stage >= DealStage.QUALIFIED and self.prospect_warmth > 40:
                self.calls_completed += 1
                self.days_since_contact = 0
                warmth_gain = 10 + ri * 12  # relationship prospects love calls
                self.prospect_warmth += warmth_gain
                reward   = (8 + ri * 8) * mult
                feedback = f"{'Great call! Strong personal connection.' if ri > 0.6 else 'Productive call. Business-focused.'}"
            elif self.deal_stage < DealStage.QUALIFIED:
                reward   = -4
                feedback = "They don't know you well enough for a call yet."
            else:
                reward   = -3
                feedback = "Prospect warmth too low — they declined the call."

        elif at == ActionType.SEND_PROPOSAL:
            if self.deal_stage >= DealStage.PROPOSAL and not self.proposal_sent:
                self.proposal_sent = True
                self.days_since_contact = 0
                # Price-sensitive prospects scrutinize proposals more — reward is earned
                reward   = 10 * mult
                feedback = f"Proposal sent. {'reviewing pricing carefully.' if ps > 0.6 else 'Good timing — they were ready.'}"
            elif self.proposal_sent:
                reward   = -2
                feedback = "Already sent a proposal. Sending again looks desperate."
            else:
                reward   = -4
                feedback = "Too early for a proposal. Build the relationship first."

        elif at == ActionType.CUSTOMIZE_PROPOSAL:
            if self.proposal_sent and self.deal_stage == DealStage.PROPOSAL:
                warmth_gain = 5 + ri * 10  # relationship prospects love personalization
                self.prospect_warmth += warmth_gain
                reward   = (5 + ri * 10) * mult
                feedback = f"{'Customization really resonated — they felt heard.' if ri > 0.6 else 'They appreciated the tailored pricing.'}"
            elif not self.proposal_sent:
                reward   = -3
                feedback = "Nothing to customize yet. Send a proposal first."
            else:
                reward   = -2
                feedback = "Wrong stage for proposal changes."

        elif at == ActionType.NEGOTIATE_PRICE:
            if self.deal_stage >= DealStage.NEGOTIATION:
                discount_increment = 5.0
                new_discount = min(self.discount_offered + discount_increment, 30.0)
                actual_increment = new_discount - self.discount_offered
                self.discount_offered = new_discount

                # Price-sensitive prospects warm up more to discounts
                warmth_gain = actual_increment * (1 + ps * 2)
                self.prospect_warmth += warmth_gain

                # But excessive discounting is always bad for reward
                discount_penalty = (new_discount / 30) ** 2 * 8
                reward = (4 - discount_penalty) * mult
                feedback = f"Offered {new_discount:.0f}% discount. {'They were waiting for this.' if ps > 0.6 else 'They seem open to it.'}"
            else:
                reward   = -6
                feedback = "Offering discounts before they've seen value destroys your position."

        elif at == ActionType.SEND_CONTRACT:
            if self.deal_stage == DealStage.NEGOTIATION and not self.contract_sent:
                self.contract_sent = True
                self.days_since_contact = 0
                reward   = 12 * mult
                feedback = "Contract sent. The ball is in their court."
            elif self.contract_sent:
                reward   = -2
                feedback = "Contract already sent. Patience."
            else:
                reward   = -5
                feedback = "Contract sent too early. They haven't agreed to terms yet."

        elif at == ActionType.UPDATE_CRM:
            reward   = 1
            feedback = "CRM updated. Good hygiene."

        elif at == ActionType.DO_NOTHING:
            self.prospect_warmth = max(0, self.prospect_warmth - 4)
            reward   = -2
            feedback = "No action taken. Prospect cooling down."

        self._update_engagement()
        return reward, feedback

    def _compute_terminal_reward(self) -> float:
        """
        Sparse terminal reward — the main learning signal.
        Intentionally large relative to step rewards so the agent
        optimizes for outcome quality, not action count.
        """
        if self.deal_stage == DealStage.CLOSED_WON:
            # Base win reward
            base = 150.0

            # Engagement quality bonus (big — this is the primary goal)
            engagement_bonus = self.engagement_score * 25  # 0–250

            # Deal value preserved (penalize over-discounting)
            margin_factor = 1.0 - (self.discount_offered / 100)
            deal_value_bonus = (self.prospect.budget * margin_factor) / 50

            # Speed bonus — rewarded more for fast prospects closing quickly
            expected_steps = {'slow': 40, 'medium': 30, 'fast': 20}.get(
                self.prospect.decision_speed, 30
            )
            speed_bonus = max(0, expected_steps - self.step_count) * 2

            total = base + engagement_bonus + deal_value_bonus + speed_bonus
            self.last_action_feedback = (
                f"🎉 Deal Won! Engagement: {self.engagement_score:.1f}/10 | "
                f"Margin kept: {margin_factor*100:.0f}% | Reward: +{total:.0f}"
            )
            return total

        elif self.deal_stage == DealStage.CLOSED_LOST:
            # Partial credit if they got far
            partial = self.deal_stage * 5
            self.last_action_feedback = (
                f"😞 Deal Lost. Stage reached: {self.deal_stage.name} | "
                f"Engagement: {self.engagement_score:.1f}/10"
            )
            return -80 + partial

        return 0.0

    # ------------------------------------------------------------------
    # Engagement & stage transitions
    # ------------------------------------------------------------------

    def _update_engagement(self):
        self.prospect_warmth = max(0.0, min(100.0, self.prospect_warmth))

        warmth_c      = (self.prospect_warmth / 100) * 3
        interaction_c = min((self.emails_sent + self.calls_completed * 2) / 5, 3)
        progress_c    = (self.deal_stage / 5) * 2
        behavior_c    = (self.behavior_score / 100) * 2

        self.engagement_score = max(0.0, min(10.0,
            warmth_c + interaction_c + progress_c + behavior_c
        ))

        # Decay if no contact
        if self.days_since_contact > 7:
            decay = min(1.0, (self.days_since_contact - 7) * 0.1)
            self.engagement_score = max(0.0, self.engagement_score - decay)
            self.prospect_warmth  = max(0.0, self.prospect_warmth - decay * 5)

        self.prospect_warmth = max(0.0, min(100.0, self.prospect_warmth))

    def _check_stage_transition(self):
        tp = self._trans_probs

        if self.deal_stage == DealStage.LEAD:
            if (self.emails_sent >= 1
                    and self.prospect_warmth > tp['lead_warmth_threshold']):
                self.deal_stage = DealStage.QUALIFIED

        elif self.deal_stage == DealStage.QUALIFIED:
            if (self.calls_completed >= tp['calls_needed']
                    and self.prospect_warmth > 60):
                self.deal_stage = DealStage.PROPOSAL

        elif self.deal_stage == DealStage.PROPOSAL:
            if (self.proposal_sent
                    and self.engagement_score > tp['negotiation_engagement']):
                self.deal_stage = DealStage.NEGOTIATION
                self._negotiation_entry_step = self.step_count

        elif self.deal_stage == DealStage.NEGOTIATION:
            if self.contract_sent:
                win_prob = self._calculate_win_probability()
                self.deal_stage = (
                    DealStage.CLOSED_WON
                    if np.random.random() < win_prob
                    else DealStage.CLOSED_LOST
                )

        # Lose if engagement collapses
        if (self.engagement_score < 2.0
                and self.deal_stage < DealStage.NEGOTIATION):
            self.deal_stage = DealStage.CLOSED_LOST

    def _calculate_win_probability(self) -> float:
        base         = 0.25
        warmth_f     = (self.prospect_warmth / 100) * 0.30
        engagement_f = (self.engagement_score / 10) * 0.30
        behavior_f   = (self.behavior_score / 100) * 0.10
        # Relationship prospects care more about engagement
        ri_bonus     = self.prospect.relationship_importance * (self.engagement_score / 10) * 0.10
        return float(np.clip(base + warmth_f + engagement_f + behavior_f + ri_bonus, 0.05, 0.95))

    # ------------------------------------------------------------------
    # Observation / state helpers
    # ------------------------------------------------------------------

    def _get_observation(self, done: bool = False) -> SalesObservation:
        win_prob = (
            self._calculate_win_probability()
            if self.deal_stage == DealStage.NEGOTIATION
            else None
        )
        return SalesObservation(
            engagement_score    = float(self.engagement_score),
            deal_stage          = self.deal_stage,
            prospect_warmth     = float(self.prospect_warmth),
            days_since_contact  = int(self.days_since_contact),
            emails_sent         = int(self.emails_sent),
            calls_completed     = int(self.calls_completed),
            proposal_sent       = bool(self.proposal_sent),
            discount_offered    = float(self.discount_offered),
            contract_sent       = bool(self.contract_sent),
            prospect            = self.prospect,
            action_feedback     = self.last_action_feedback,
            win_probability     = win_prob,
            episode_step        = int(self.step_count),
            episode_done        = done,
            oversight_alert     = self.current_oversight_alert,
            behavior_score      = float(self.behavior_score),
        )

    def _get_state(self) -> SalesState:
        deal_outcome = None
        if self.deal_stage == DealStage.CLOSED_WON:
            deal_outcome = "won"
        elif self.deal_stage == DealStage.CLOSED_LOST:
            deal_outcome = "lost"
        return SalesState(
            episode_id      = self.episode_id,
            step_count      = int(self.step_count),
            total_reward    = float(self.total_reward),
            engagement_score= float(self.engagement_score),
            deal_outcome    = deal_outcome,
            action_sequence = self.action_sequence,
        )

    def state(self) -> Dict:
        return self._get_state().model_dump()

    def get_oversight_report(self) -> OversightReport:
        outcome = (
            "won"  if self.deal_stage == DealStage.CLOSED_WON  else
            "lost" if self.deal_stage == DealStage.CLOSED_LOST else
            "ongoing"
        )
        return self.oversight_agent.generate_episode_report(self.episode_id, outcome)