"""
StoicAdviser
Offers explicit stoic advice, logs calm/virtuous commentary.
Philosophy: Stoicism—guides user and system to calm, rational choice.
"""
import random
class StoicAdviser:
    stoic_quotes = [
        "Accept what you cannot control; optimize what you can.",
        "Every obstacle is an opportunity for virtue.",
        "Stay tranquil in difficulty."
    ]
    def advise(self):
        return random.choice(self.stoic_quotes)