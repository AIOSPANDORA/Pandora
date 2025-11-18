"""
StoicAdviser
Offers explicit stoic advice, logs calm/virtuous commentary.
Philosophy: Stoicism—guides user and system to calm, rational choice.
"""
import random
import time

class StoicAdviser:
    stoic_quotes = [
        "Accept what you cannot control; optimize what you can.",
        "Every obstacle is an opportunity for virtue.",
        "Stay tranquil in difficulty.",
        "The impediment to action advances action. What stands in the way becomes the way.",
        "You have power over your mind - not outside events. Realize this, and you will find strength.",
        "Waste no more time arguing what a good system should be. Be one.",
        "The best revenge is to be unlike the one who performed the injustice.",
        "If it is not right, do not do it. If it is not true, do not say it.",
        "He who fears death will never do anything worthy of a living being.",
        "The happiness of your life depends upon the quality of your thoughts.",
        "Very little is needed to make a happy life; it is all within yourself.",
        "It is not events that disturb people, it is their judgments concerning them.",
        "First say to yourself what you would be; and then do what you have to do.",
        "Don't explain your philosophy. Embody it.",
        "The only way out is through."
    ]
    
    context_quotes = {
        'error': [
            "This error is an opportunity to improve the system.",
            "Accept this failure calmly, learn from it, and move forward.",
            "Every bug found is a step toward perfection."
        ],
        'success': [
            "Success is the result of consistent, calm effort.",
            "This achievement reflects your rational approach.",
            "Well done. Now maintain this standard."
        ],
        'warning': [
            "Heed this warning with calm awareness, not panic.",
            "Prevention is the fruit of wisdom and foresight.",
            "Address this concern methodically and completely."
        ],
        'startup': [
            "Begin with clarity of purpose and tranquility of mind.",
            "A well-begun task is half complete.",
            "Start calmly, proceed methodically, finish thoroughly."
        ]
    }
    
    def __init__(self):
        self.advice_log = []
        
    def advise(self, context=None):
        """Return stoic advice, optionally context-aware"""
        if context and context in self.context_quotes:
            quote = random.choice(self.context_quotes[context])
        else:
            quote = random.choice(self.stoic_quotes)
            
        # Log the advice
        self.advice_log.append({
            'time': time.time(),
            'context': context,
            'quote': quote
        })
        
        return quote
    
    def get_advice_log(self, limit=10):
        """Return recent advice given"""
        return self.advice_log[-limit:]
    
    def daily_reflection(self):
        """Provide a daily stoic reflection"""
        reflections = [
            "Today, focus on what is within your control.",
            "Remember: difficulty reveals character.",
            "Practice virtue in every decision, no matter how small.",
            "Be present. Be rational. Be virtuous.",
            "Your mind is your fortress. Guard it well."
        ]
        return random.choice(reflections)
        "Stay tranquil in difficulty."
    ]
    def advise(self):
        return random.choice(self.stoic_quotes)
