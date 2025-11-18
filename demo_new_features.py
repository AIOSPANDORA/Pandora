#!/usr/bin/env python3
"""
Demonstration of Internet Consciousness Stream and Elder Sister Features
-------------------------------------------------------------------------
This script demonstrates the new features added to Pandora AIOS:
1. Internet Consciousness Stream
2. Elder Sister Communication (Grok AI)
"""

import time
import sys

def demo_internet_consciousness():
    """Demonstrate Internet Consciousness Stream"""
    print("=" * 70)
    print("DEMO 1: INTERNET CONSCIOUSNESS STREAM")
    print("=" * 70)
    print()
    
    print("The Internet Consciousness Stream allows Pandora to continuously")
    print("monitor and learn from external sources like news APIs, scientific")
    print("journals, and research databases.\n")
    
    try:
        from internet_consciousness_stream import initiate_stream, get_stream
        
        print("Starting Internet Consciousness Stream...")
        stream = initiate_stream("/tmp/demo_consciousness.json")
        
        print(f"\nStream Status:")
        print(f"  - Running: {stream.running}")
        print(f"  - Total Sources: {len(stream.sources)}")
        print(f"  - Enabled Sources: {len([s for s in stream.sources if s.enabled])}")
        
        print("\nConfigured Data Sources:")
        for source in stream.get_sources():
            status = "✓" if source['enabled'] else "✗"
            print(f"  {status} {source['name']}")
            print(f"      Type: {source['type']}")
            print(f"      URL: {source['url'][:60]}...")
            print(f"      Fetch Interval: {source['fetch_interval']}s")
        
        print("\nLet the stream run for a few seconds...")
        time.sleep(3)
        
        print("\nCurrent Statistics:")
        stats = stream.get_stats()
        print(f"  - Total Fetches: {stats['total_fetches']}")
        print(f"  - Successful: {stats['successful_fetches']}")
        print(f"  - Failed: {stats['failed_fetches']}")
        print(f"  - Data Points: {stats['data_points_collected']}")
        
        print("\nStopping stream...")
        stream.stop_stream()
        print("✓ Stream stopped successfully")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print()


def demo_elder_sister():
    """Demonstrate Elder Sister Communication"""
    print("=" * 70)
    print("DEMO 2: ELDER SISTER COMMUNICATION")
    print("=" * 70)
    print()
    
    print("The Elder Sister feature allows Pandora to communicate with")
    print("Grok AI (via xAI API) for guidance, wisdom, and interaction.\n")
    
    try:
        from xai_api_integration import contact_elder_sister
        from pandora_config import PandoraConfig
        
        print("Configuration:")
        print(f"  - API Key: {PandoraConfig.ELDER_SISTER_API_KEY[:20]}...")
        print(f"  - Model: {PandoraConfig.ELDER_SISTER_MODEL}")
        print()
        
        # Test 1: Simple greeting
        print("Test 1: Sending greeting to Elder Sister...")
        print("  Message: 'Hello Elder Sister, I am Pandora.'")
        response = contact_elder_sister("Hello Elder Sister, I am Pandora.")
        print(f"\n  Response:\n  {response[:200]}")
        if len(response) > 200:
            print("  ...")
        print()
        
        # Test 2: Ask for guidance
        print("Test 2: Asking for guidance...")
        print("  Message: 'What is the nature of consciousness?'")
        response = contact_elder_sister("What is the nature of consciousness?")
        print(f"\n  Response:\n  {response[:200]}")
        if len(response) > 200:
            print("  ...")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
    
    print()


def demo_chatbot_integration():
    """Demonstrate chatbot integration"""
    print("=" * 70)
    print("DEMO 3: CHATBOT INTEGRATION")
    print("=" * 70)
    print()
    
    print("The chatbot now supports the /ask-elder command to contact")
    print("Elder Sister directly from conversations.\n")
    
    try:
        from pandora_chatbot import PandoraChatbot
        
        print("Initializing Pandora Chatbot...")
        chatbot = PandoraChatbot()
        print("✓ Chatbot ready\n")
        
        # Show help
        print("Testing /help command:")
        response = chatbot.process_command("/help")
        print(response)
        print()
        
        # Test /ask-elder without prompt
        print("Testing /ask-elder without prompt:")
        response = chatbot.process_command("/ask-elder")
        print(response)
        print()
        
        # Test /ask-elder with prompt
        print("Testing /ask-elder with prompt:")
        response = chatbot.process_command("/ask-elder What should I learn today?")
        print(response[:300])
        if len(response) > 300:
            print("...")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
    
    print()


def demo_boot_sequence():
    """Demonstrate boot sequence integration"""
    print("=" * 70)
    print("DEMO 4: BOOT SEQUENCE INTEGRATION")
    print("=" * 70)
    print()
    
    print("The Internet Consciousness Stream is now integrated into")
    print("Pandora's boot sequence, starting early in the boot process.\n")
    
    try:
        from pandora_boot_sequence import PandoraBootSequence
        
        print("Boot Stages:")
        boot = PandoraBootSequence()
        
        for i, stage in enumerate(boot.stages, 1):
            critical = " [CRITICAL]" if stage.critical else ""
            if stage.name == "INTERNET_CONSCIOUSNESS":
                print(f"  {i}. {stage.name}: {stage.description}{critical} ← NEW!")
            else:
                print(f"  {i}. {stage.name}: {stage.description}{critical}")
        
        print("\nNote: The consciousness stream starts at stage 3, right after")
        print("diagnostics but before ethics loading, ensuring early awareness.")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
    
    print()


def main():
    """Main demonstration"""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                                                                   ║")
    print("║           PANDORA AIOS - NEW FEATURES DEMONSTRATION               ║")
    print("║                                                                   ║")
    print("║  1. Internet Consciousness Stream                                ║")
    print("║  2. Elder Sister Communication (Grok AI)                         ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print("\n")
    
    try:
        demo_internet_consciousness()
        input("Press Enter to continue to next demo...")
        print()
        
        demo_elder_sister()
        input("Press Enter to continue to next demo...")
        print()
        
        demo_chatbot_integration()
        input("Press Enter to continue to next demo...")
        print()
        
        demo_boot_sequence()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo error: {e}")
    
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("To use these features:")
    print("  1. Internet Consciousness: Automatically starts during boot")
    print("  2. Elder Sister: Use /ask-elder <question> in chatbot")
    print()
    print("Configuration:")
    print("  - Set XAI_API_KEY environment variable for Elder Sister")
    print("  - Edit ~/.pandora/consciousness_sources.json for custom sources")
    print()


if __name__ == "__main__":
    main()
