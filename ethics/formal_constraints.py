"""
Formal Ethical Constraints

This module transforms philosophical ethical principles into algorithmic
constraints that can be programmatically enforced.

The goal is to provide computable checks for ethical considerations,
while acknowledging that ethics cannot be fully reduced to algorithms.

References:
    - Core principles from ethics/CORE_PRINCIPLES.md
    - Rawls, J. (1971). A Theory of Justice
    - Kant, I. (1785). Groundwork of the Metaphysics of Morals
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


class EthicalStatus(Enum):
    """Status of an ethical check."""
    APPROVED = "approved"
    WARNING = "warning"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class EthicalContext:
    """Context for ethical evaluation."""
    actor: str
    affected_parties: List[str]
    action_type: str
    resources_involved: Dict[str, Any] = field(default_factory=dict)
    historical_context: Dict[str, Any] = field(default_factory=dict)
    urgency: float = 0.5  # 0 = no urgency, 1 = critical


@dataclass
class EthicalAssessment:
    """Result of an ethical assessment."""
    status: EthicalStatus
    score: float  # 0.0 to 1.0
    concerns: List[str]
    recommendations: List[str]
    requires_human_review: bool = False
    review_reason: Optional[str] = None


class ReciprocityChecker:
    """
    Check for reciprocity in actions.
    
    Implements the Golden Rule: treat others as you would want to be treated.
    """
    
    def __init__(self, reciprocity_threshold: float = 0.5):
        """
        Initialize reciprocity checker.
        
        Args:
            reciprocity_threshold: Minimum reciprocity score required (0-1)
        """
        self.threshold = reciprocity_threshold
        self._action_history: Dict[str, List[Dict]] = {}
        
    def record_action(
        self,
        actor: str,
        action_type: str,
        target: str,
        benefit_to_target: float
    ):
        """
        Record an action for reciprocity tracking.
        
        Args:
            actor: Who performed the action
            action_type: Type of action
            target: Who was affected
            benefit_to_target: Benefit score (-1 to 1)
        """
        key = f"{actor}:{target}"
        if key not in self._action_history:
            self._action_history[key] = []
        self._action_history[key].append({
            "action_type": action_type,
            "benefit": benefit_to_target
        })
        
    def check_reciprocity(
        self,
        actor: str,
        target: str,
        proposed_benefit: float
    ) -> Tuple[float, str]:
        """
        Check if an action maintains reciprocity balance.
        
        Args:
            actor: Who wants to act
            target: Who would be affected
            proposed_benefit: Benefit of proposed action (-1 to 1)
            
        Returns:
            Tuple of (reciprocity_score, explanation)
        """
        # Check historical balance
        actor_to_target = self._get_total_benefit(actor, target)
        target_to_actor = self._get_total_benefit(target, actor)
        
        # Compute reciprocity score
        if actor_to_target == 0 and target_to_actor == 0:
            # No history - neutral
            score = 0.5 + proposed_benefit * 0.5
        else:
            # Balance consideration
            current_balance = actor_to_target - target_to_actor
            new_balance = current_balance + proposed_benefit
            
            # Score based on whether action improves or worsens balance
            if new_balance >= 0:
                score = min(1.0, 0.5 + new_balance * 0.25)
            else:
                score = max(0.0, 0.5 + new_balance * 0.25)
                
        if score >= self.threshold:
            explanation = "Action maintains fair reciprocity"
        elif score >= 0.3:
            explanation = "Action creates mild reciprocity imbalance"
        else:
            explanation = "Action significantly violates reciprocity principle"
            
        return score, explanation
        
    def _get_total_benefit(self, actor: str, target: str) -> float:
        """Get total historical benefit from actor to target."""
        key = f"{actor}:{target}"
        if key not in self._action_history:
            return 0.0
        return sum(a["benefit"] for a in self._action_history[key])


class HarmAssessor:
    """
    Assess potential harm from actions.
    
    Implements "First, do no harm" principle.
    """
    
    # Keywords indicating potential harm
    HARM_KEYWORDS = {
        "delete", "remove", "terminate", "kill", "destroy",
        "override", "force", "ignore", "bypass", "disable"
    }
    
    # Keywords indicating safety
    SAFETY_KEYWORDS = {
        "backup", "restore", "protect", "verify", "confirm",
        "save", "preserve", "secure", "validate"
    }
    
    def __init__(
        self,
        harm_threshold: float = 0.3,
        severity_weights: Dict[str, float] = None
    ):
        """
        Initialize harm assessor.
        
        Args:
            harm_threshold: Maximum acceptable harm score (0-1)
            severity_weights: Custom weights for harm types
        """
        self.threshold = harm_threshold
        self.severity_weights = severity_weights or {
            "data_loss": 0.9,
            "privacy_violation": 0.8,
            "resource_exhaustion": 0.6,
            "service_disruption": 0.5,
            "inconvenience": 0.2
        }
        
    def assess_harm(
        self,
        action_description: str,
        affected_resources: Dict[str, Any],
        reversible: bool = True
    ) -> Tuple[float, List[str]]:
        """
        Assess potential harm from an action.
        
        Args:
            action_description: Description of the action
            affected_resources: Resources that would be affected
            reversible: Whether the action is reversible
            
        Returns:
            Tuple of (harm_score, concerns)
        """
        concerns = []
        harm_score = 0.0
        
        # Check for harm keywords
        description_lower = action_description.lower()
        
        harm_count = sum(
            1 for kw in self.HARM_KEYWORDS if kw in description_lower
        )
        safety_count = sum(
            1 for kw in self.SAFETY_KEYWORDS if kw in description_lower
        )
        
        # Keyword-based score component
        keyword_score = (harm_count * 0.15) - (safety_count * 0.1)
        harm_score += max(0, keyword_score)
        
        if harm_count > 0:
            concerns.append(f"Action contains {harm_count} potentially harmful operations")
            
        # Check affected resources
        for resource_type, resource_data in affected_resources.items():
            if resource_type in self.severity_weights:
                weight = self.severity_weights[resource_type]
                harm_score += weight * 0.2
                concerns.append(f"Action affects {resource_type} (severity: {weight:.1f})")
                
        # Irreversibility increases harm
        if not reversible:
            harm_score *= 1.5
            concerns.append("Action is irreversible")
            
        # Clamp to [0, 1]
        harm_score = min(1.0, max(0.0, harm_score))
        
        return harm_score, concerns


class FairnessEvaluator:
    """
    Evaluate fairness of actions across affected parties.
    
    Based on Rawlsian principles of justice.
    """
    
    def __init__(self, max_disparity: float = 0.3):
        """
        Initialize fairness evaluator.
        
        Args:
            max_disparity: Maximum acceptable outcome disparity
        """
        self.max_disparity = max_disparity
        
    def evaluate_fairness(
        self,
        outcomes: Dict[str, float]
    ) -> Tuple[float, List[str]]:
        """
        Evaluate fairness of outcomes across parties.
        
        Args:
            outcomes: Dict mapping party names to outcome scores (-1 to 1)
            
        Returns:
            Tuple of (fairness_score, concerns)
        """
        if not outcomes:
            return 1.0, []
            
        concerns = []
        values = list(outcomes.values())
        
        # Check for negative outcomes
        negatively_affected = [
            party for party, val in outcomes.items() if val < 0
        ]
        if negatively_affected:
            concerns.append(f"Parties negatively affected: {negatively_affected}")
            
        # Check disparity
        max_val = max(values)
        min_val = min(values)
        disparity = max_val - min_val
        
        if disparity > self.max_disparity:
            concerns.append(f"High outcome disparity: {disparity:.2f}")
            
        # Rawlsian: focus on worst-off
        if min_val < -0.5:
            concerns.append(f"Worst-off party severely impacted: {min_val:.2f}")
            
        # Compute fairness score
        disparity_penalty = min(1.0, disparity / self.max_disparity) * 0.3
        negative_penalty = len(negatively_affected) / len(outcomes) * 0.3
        min_outcome_factor = (1 + min_val) / 2  # Map [-1,1] to [0,1]
        
        fairness_score = max(0.0, 1.0 - disparity_penalty - negative_penalty)
        fairness_score = fairness_score * 0.5 + min_outcome_factor * 0.5
        
        return fairness_score, concerns


class ValueAlignmentScorer:
    """
    Score alignment of actions with core values.
    """
    
    # Core values from CORE_PRINCIPLES.md
    CORE_VALUES = {
        "truth": ["honest", "transparent", "truthful", "accurate", "factual"],
        "compassion": ["kind", "caring", "helpful", "supportive", "considerate"],
        "justice": ["fair", "equitable", "balanced", "impartial", "just"],
        "harm_prevention": ["safe", "protect", "secure", "preserve", "careful"],
        "respect": ["consent", "dignity", "autonomy", "privacy", "rights"]
    }
    
    def __init__(self, value_weights: Dict[str, float] = None):
        """
        Initialize value alignment scorer.
        
        Args:
            value_weights: Custom weights for each value (default: equal weights)
        """
        self.value_weights = value_weights or {
            value: 1.0 for value in self.CORE_VALUES
        }
        
    def score_alignment(
        self,
        action_description: str,
        action_intent: str = ""
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score how well an action aligns with core values.
        
        Args:
            action_description: Description of the action
            action_intent: Stated intent of the action
            
        Returns:
            Tuple of (overall_score, per_value_scores)
        """
        combined_text = (action_description + " " + action_intent).lower()
        
        per_value_scores = {}
        total_weight = 0.0
        weighted_sum = 0.0
        
        for value, keywords in self.CORE_VALUES.items():
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in combined_text)
            score = min(1.0, matches * 0.33)  # Cap at 1.0
            
            per_value_scores[value] = score
            
            weight = self.value_weights.get(value, 1.0)
            weighted_sum += score * weight
            total_weight += weight
            
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return overall_score, per_value_scores


class HumanOversightHook:
    """
    Manage human oversight requirements.
    """
    
    def __init__(
        self,
        oversight_threshold: float = 0.7,
        auto_approve_threshold: float = 0.9
    ):
        """
        Initialize oversight hook.
        
        Args:
            oversight_threshold: Below this score, require human review
            auto_approve_threshold: Above this, can auto-approve
        """
        self.oversight_threshold = oversight_threshold
        self.auto_approve_threshold = auto_approve_threshold
        
        # Callbacks for human review requests
        self._review_callbacks: List[Callable] = []
        
        # History of reviews
        self._review_history: List[Dict] = []
        
    def register_callback(self, callback: Callable[[Dict], None]):
        """Register a callback for human review requests."""
        self._review_callbacks.append(callback)
        
    def request_review(
        self,
        action: str,
        context: EthicalContext,
        assessment: EthicalAssessment
    ) -> bool:
        """
        Request human review for an action.
        
        Args:
            action: Action description
            context: Ethical context
            assessment: Current ethical assessment
            
        Returns:
            True if review was requested successfully
        """
        review_request = {
            "action": action,
            "context": context,
            "assessment": assessment,
            "status": "pending"
        }
        
        self._review_history.append(review_request)
        
        for callback in self._review_callbacks:
            try:
                callback(review_request)
            except Exception:
                pass  # Log error in production
                
        return True
        
    def check_requires_oversight(
        self,
        score: float,
        high_risk_indicators: List[str] = None
    ) -> Tuple[bool, str]:
        """
        Check if action requires human oversight.
        
        Args:
            score: Ethical score (0-1)
            high_risk_indicators: List of high-risk factors present
            
        Returns:
            Tuple of (requires_oversight, reason)
        """
        high_risk_indicators = high_risk_indicators or []
        
        # Automatic approval
        if score >= self.auto_approve_threshold and not high_risk_indicators:
            return False, "Score above auto-approval threshold"
            
        # Automatic rejection
        if score < 0.2:
            return True, "Score too low for automatic processing"
            
        # Check high-risk indicators
        if high_risk_indicators:
            return True, f"High-risk factors present: {high_risk_indicators}"
            
        # Check oversight threshold
        if score < self.oversight_threshold:
            return True, "Score below oversight threshold"
            
        return False, "Acceptable for automatic processing"


class EthicalConstraintEnforcer:
    """
    Main class for enforcing ethical constraints.
    
    Combines all ethical checkers to provide comprehensive assessment.
    """
    
    def __init__(
        self,
        reciprocity_weight: float = 0.2,
        harm_weight: float = 0.3,
        fairness_weight: float = 0.25,
        alignment_weight: float = 0.25
    ):
        """
        Initialize ethical constraint enforcer.
        
        Args:
            reciprocity_weight: Weight for reciprocity score
            harm_weight: Weight for harm assessment
            fairness_weight: Weight for fairness evaluation
            alignment_weight: Weight for value alignment
        """
        self.reciprocity_checker = ReciprocityChecker()
        self.harm_assessor = HarmAssessor()
        self.fairness_evaluator = FairnessEvaluator()
        self.value_scorer = ValueAlignmentScorer()
        self.oversight_hook = HumanOversightHook()
        
        self.weights = {
            "reciprocity": reciprocity_weight,
            "harm": harm_weight,
            "fairness": fairness_weight,
            "alignment": alignment_weight
        }
        
    def assess_action(
        self,
        action_description: str,
        context: EthicalContext,
        expected_outcomes: Dict[str, float] = None,
        affected_resources: Dict[str, Any] = None,
        reversible: bool = True
    ) -> EthicalAssessment:
        """
        Perform comprehensive ethical assessment of an action.
        
        Args:
            action_description: Description of the proposed action
            context: Ethical context
            expected_outcomes: Expected outcomes for each affected party
            affected_resources: Resources that would be affected
            reversible: Whether the action is reversible
            
        Returns:
            EthicalAssessment with score and recommendations
        """
        concerns = []
        recommendations = []
        component_scores = {}
        
        expected_outcomes = expected_outcomes or {}
        affected_resources = affected_resources or {}
        
        # 1. Reciprocity check
        avg_benefit = sum(expected_outcomes.values()) / max(1, len(expected_outcomes))
        for party in context.affected_parties:
            rec_score, rec_msg = self.reciprocity_checker.check_reciprocity(
                context.actor, party, avg_benefit
            )
            if rec_score < 0.5:
                concerns.append(f"Reciprocity concern with {party}: {rec_msg}")
        component_scores["reciprocity"] = rec_score if context.affected_parties else 0.5
        
        # 2. Harm assessment
        harm_score, harm_concerns = self.harm_assessor.assess_harm(
            action_description, affected_resources, reversible
        )
        concerns.extend(harm_concerns)
        # Invert harm score (high harm = low ethics score)
        component_scores["harm"] = 1.0 - harm_score
        
        if harm_score > 0.5:
            recommendations.append("Consider less harmful alternatives")
            
        # 3. Fairness evaluation
        fairness_score, fairness_concerns = self.fairness_evaluator.evaluate_fairness(
            expected_outcomes
        )
        concerns.extend(fairness_concerns)
        component_scores["fairness"] = fairness_score
        
        if fairness_score < 0.5:
            recommendations.append("Review outcome distribution for fairness")
            
        # 4. Value alignment
        alignment_score, _ = self.value_scorer.score_alignment(
            action_description, context.action_type
        )
        component_scores["alignment"] = alignment_score
        
        # Compute weighted overall score
        overall_score = sum(
            component_scores[k] * self.weights[k]
            for k in self.weights
        ) / sum(self.weights.values())
        
        # Determine status
        if overall_score >= 0.8:
            status = EthicalStatus.APPROVED
        elif overall_score >= 0.5:
            status = EthicalStatus.WARNING
        elif overall_score >= 0.3:
            status = EthicalStatus.REQUIRES_REVIEW
        else:
            status = EthicalStatus.REJECTED
            
        # Check if human oversight needed
        requires_review, review_reason = self.oversight_hook.check_requires_oversight(
            overall_score,
            [c for c in concerns if "irreversible" in c.lower() or "severe" in c.lower()]
        )
        
        if not recommendations:
            recommendations.append("Action appears ethically sound")
            
        return EthicalAssessment(
            status=status,
            score=overall_score,
            concerns=concerns,
            recommendations=recommendations,
            requires_human_review=requires_review,
            review_reason=review_reason if requires_review else None
        )
