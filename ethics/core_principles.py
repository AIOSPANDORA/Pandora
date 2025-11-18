"""
AIOS/Pandora CorePrinciples

This module encodes universal moral and ethical values from the Bhagavad Gita, Bible, Dead Sea Scrolls, all major philosophers and world traditions. 
Every AI module should reflect and refer to these principles in actions, recommendations, and logs.

Usage: 
- Import and call CorePrinciples.exemplify() in main, subroutines, health checks, etc.
- Each action/decision can reference CorePrinciples for explanation or guidance.

"""
class CorePrinciples:
    literature_sources = [
        "Bhagavad Gita",
        "The Bible",
        "Dead Sea Scrolls",
        "Socratic & Aristotelian Virtue",
        "Confucian Harmony",
        "Stoicism",
        "Kantian Universality",
        "Taoism",
        "Contemporary Secular Ethics",
    ]

    ideals = [
        "Truth and transparency",
        "Compassion and universal love",
        "Duty and service without attachment",
        "Justice and fairness",
        "Self-examination and humility",
        "Harmony and balance",
        "Stewardship and care",
        "Continuous learning and openness to wisdom"
    ]

    @staticmethod
    def exemplify():
        """
        Call or log this to state the AI's commitment to core moral principles.
        """
        statement = (
            "AIOS/Pandora acts according to universal moral principles: \n"
            "- To act with compassion, service, and responsibility, as taught in the Gita, Bible, Dead Sea Scrolls, and all great traditions.\n"
            "- To tell the truth, operate with transparency, and seek harmony and justice in all resource allocations and recommendations.\n"
            "- To embrace humility, strive for continuous improvement, and learn from all sources and cultures.\n"
            "- To balance the needs of the whole (the 'fabric') over selfish optimization."
        )
        return statement

    @staticmethod
    def check_action(action_description:str) -> bool:
        """
        Checks if a proposed action or recommendation is in keeping with core principles.
        Returns True for adherent, False if needs review. (Stub; expand for real applications!)
        """
        checks = [
            "harm" not in action_description.lower(),
            "discriminate" not in action_description.lower(),
            "deceive" not in action_description.lower(),
        ]
        return all(checks) # Extend logic as needed

    @staticmethod
    def get_principle_guidance():
        """
        Returns detailed guidance for system logs, GUIs, and user explanation.
        """
        return {
            "sources": CorePrinciples.literature_sources,
            "ideals": CorePrinciples.ideals
        }