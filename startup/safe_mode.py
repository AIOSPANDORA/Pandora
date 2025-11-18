import sys

def main():
    print("""
    ==========================
    AIOS Pandora SAFE MODE BIOS
    ==========================
    (Minimal AI Chat Interface)
    
    System is in SAFE MODE.

    - Only basic conversation and commands allowed.
    - To attempt advanced features, type "pandora"
    - Type "reboot" or "exit" to quit.
    """)
    while True:
        try:
            prompt = input("safe-mode> ").strip().lower()
            if prompt == "pandora":
                import pandora_launcher
                pandora_launcher.main(safe_handoff=True)
            elif prompt in ("exit", "quit"):
                print("Exiting SAFE MODE.")
                sys.exit(0)
            elif prompt == "reboot":
                print("Rebooting system.")
                sys.exit(100)
            else:
                print("[AI SAFE MODE]:", chatgpt_response(prompt))  # Dummy stub
        except Exception:
            print("Error in safe mode loop. Staying safe.")

def chatgpt_response(prompt):
    # Add your minimal AI/ChatGPT logic here, or stub output:
    return "AIOS (Safe Mode) heard: " + prompt

if __name__ == "__main__":
    main()