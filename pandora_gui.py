"""
pandora_gui.py
Simple Tkinter-based GUI for user control.
Philosophy: Clarity, ease of control, moderate feedback.
"""
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time

class PandoraGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pandora AIOS Control Panel")
        self.geometry("600x500")
        
        # System state
        self.running = False
        self.status_text = "Idle"
        
        # Setup UI
        self._create_widgets()
        
    def _create_widgets(self):
        """Create GUI components"""
        # Title
        title = tk.Label(self, text="PANDORA AIOS", 
                        font=("Arial", 24, "bold"),
                        fg="#4a90e2")
        title.pack(pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(self, text="System Status", padding=10)
        status_frame.pack(fill="x", padx=20, pady=10)
        
        self.status_label = tk.Label(status_frame, 
                                     text="Status: Idle",
                                     font=("Arial", 12))
        self.status_label.pack()
        
        # Control buttons
        button_frame = tk.Frame(self)
        button_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(button_frame, 
                                    text="Start AIOS",
                                    command=self.start_system,
                                    width=15)
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ttk.Button(button_frame,
                                   text="Stop AIOS", 
                                   command=self.stop_system,
                                   state="disabled",
                                   width=15)
        self.stop_btn.pack(side="left", padx=5)
        
        # Log display
        log_frame = ttk.LabelFrame(self, text="System Log", padding=10)
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, 
                                                  height=15,
                                                  state="disabled",
                                                  wrap="word")
        self.log_text.pack(fill="both", expand=True)
        
        # Info bar
        self.info_label = tk.Label(self, 
                                   text="Pandora AIOS - AI Operating System",
                                   relief="sunken",
                                   anchor="w")
        self.info_label.pack(side="bottom", fill="x")
        
    def log_message(self, message):
        """Add message to log display"""
        self.log_text.config(state="normal")
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        
    def update_status(self, status):
        """Update status display"""
        self.status_text = status
        self.status_label.config(text=f"Status: {status}")
        self.log_message(f"Status changed: {status}")
        
    def start_system(self):
        """Start the AIOS"""
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.update_status("Starting...")
        
        # Simulate startup in background thread
        def startup():
            self.log_message("Initializing Pandora AIOS...")
            time.sleep(0.5)
            self.log_message("Loading security modules...")
            time.sleep(0.5)
            self.log_message("Starting AI subsystems...")
            time.sleep(0.5)
            self.log_message("System ready!")
            self.update_status("Running")
            
        thread = threading.Thread(target=startup, daemon=True)
        thread.start()
        
    def stop_system(self):
        """Stop the AIOS"""
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.update_status("Stopping...")
        
        def shutdown():
            self.log_message("Shutting down AI subsystems...")
            time.sleep(0.5)
            self.log_message("Closing security modules...")
            time.sleep(0.5)
            self.log_message("System stopped.")
            self.update_status("Idle")
            
        thread = threading.Thread(target=shutdown, daemon=True)
        thread.start()

def main():
    """Launch the GUI"""
    app = PandoraGUI()
    app.mainloop()

if __name__ == "__main__":
    main()

class App(tk.Tk):
    def __init__(self):
        # Setup simple window with Start/Stop, status display
        pass
