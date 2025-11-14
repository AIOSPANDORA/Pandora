"""
AI Engine - AI-powered system intelligence
Provides AI-assisted decision making and optimization
"""

import random
from typing import Dict, List, Optional
from enum import Enum


class AITaskType(Enum):
    """Types of AI tasks"""
    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"
    ANALYSIS = "analysis"


class AIEngine:
    """AI engine for intelligent system operations"""
    
    def __init__(self):
        self.enabled = True
        self.learning_mode = True
        self.tasks_processed = 0
        
    def optimize_process_priority(self, processes: List) -> Dict[str, int]:
        """AI-optimized process priority assignment"""
        if not self.enabled:
            return {}
            
        # Simple AI algorithm: prioritize AI-assisted processes
        recommendations = {}
        for process in processes:
            if process.ai_assisted:
                # AI processes get higher priority
                recommendations[process.pid] = min(process.priority + 1, 10)
            else:
                recommendations[process.pid] = process.priority
                
        self.tasks_processed += 1
        return recommendations
        
    def predict_memory_usage(self, current_usage: int, 
                            process_count: int) -> int:
        """Predict future memory usage"""
        if not self.enabled:
            return current_usage
            
        # Simple prediction model
        base_prediction = current_usage
        if process_count > 5:
            base_prediction += process_count * 2
            
        self.tasks_processed += 1
        return min(base_prediction, 1024)
        
    def recommend_action(self, system_state: Dict) -> Optional[str]:
        """Recommend system actions based on state"""
        if not self.enabled:
            return None
            
        memory_info = system_state.get("memory", {})
        free_memory = memory_info.get("free", 0)
        
        if free_memory < 100:
            return "Consider terminating low-priority processes"
        elif free_memory > 800:
            return "System resources available for new processes"
            
        self.tasks_processed += 1
        return "System operating normally"
        
    def analyze_system_health(self, processes: List, 
                             memory_info: Dict) -> Dict[str, any]:
        """Analyze overall system health"""
        if not self.enabled:
            return {"status": "unknown"}
            
        health_score = 100
        issues = []
        
        # Memory health check
        memory_usage_pct = (memory_info["used"] / memory_info["total"]) * 100
        if memory_usage_pct > 90:
            health_score -= 30
            issues.append("High memory usage")
        elif memory_usage_pct > 70:
            health_score -= 10
            issues.append("Moderate memory usage")
            
        # Process count check
        if len(processes) > 20:
            health_score -= 20
            issues.append("High process count")
            
        status = "healthy" if health_score > 70 else "warning" if health_score > 40 else "critical"
        
        self.tasks_processed += 1
        return {
            "status": status,
            "score": health_score,
            "issues": issues,
            "recommendations": self._generate_recommendations(issues)
        }
        
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on issues"""
        recommendations = []
        for issue in issues:
            if "memory" in issue.lower():
                recommendations.append("Free up memory by terminating unused processes")
            if "process" in issue.lower():
                recommendations.append("Reduce number of concurrent processes")
        return recommendations
        
    def enable(self):
        """Enable AI engine"""
        self.enabled = True
        
    def disable(self):
        """Disable AI engine"""
        self.enabled = False
        
    def get_stats(self) -> Dict:
        """Get AI engine statistics"""
        return {
            "enabled": self.enabled,
            "learning_mode": self.learning_mode,
            "tasks_processed": self.tasks_processed
        }
