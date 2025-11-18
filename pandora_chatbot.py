"""
Pandora AIOS Offline Chatbot Interface
---------------------------------------
Full GPT-style chatbot running completely offline without APIs.
Integrates all Pandora AIOS systems with local LLM models.

Features:
- Offline LLM (GPT4All, LLaMA, Mistral, etc.)
- Multi-turn conversations
- System integration (diagnostics, WSL, quantum, etc.)
- Context-aware responses
- Memory and history
- Command execution
- Research database queries
- Multi-modal support

Philosophy: Complete autonomy, privacy-first, no cloud dependencies
"""

import os
import sys
import json
import time
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict

# Try to import local LLM libraries
LLMS_AVAILABLE = {}

try:
    from gpt4all import GPT4All
    LLMS_AVAILABLE['gpt4all'] = True
except ImportError:
    LLMS_AVAILABLE['gpt4all'] = False

try:
    from llama_cpp import Llama
    LLMS_AVAILABLE['llama_cpp'] = True
except ImportError:
    LLMS_AVAILABLE['llama_cpp'] = False

try:
    import transformers
    LLMS_AVAILABLE['transformers'] = True
except ImportError:
    LLMS_AVAILABLE['transformers'] = False

# Import Pandora systems
try:
    from diagnostic_system import PandoraDiagnostics
    from universal_compatibility import CompatibilityLayer
    from quantum_overlay_profiles import QuantumOverlayManager, OverlayType
    from scientific_research_tracker import ResearchDatabase
    from pandora_knowledge_base import PandoraKnowledgeBase
    PANDORA_SYSTEMS_AVAILABLE = True
except ImportError:
    PANDORA_SYSTEMS_AVAILABLE = False
    print("[WARNING] Some Pandora systems not available")

@dataclass
class Message:
    """Represents a chat message"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class Conversation:
    """Represents a conversation thread"""
    conversation_id: str
    title: str
    messages: List[Message] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: Message):
        """Add message to conversation"""
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['messages'] = [m.to_dict() for m in self.messages]
        return data


class LocalLLM:
    """Wrapper for local LLM models"""
    
    def __init__(self, model_name: str = "auto", model_path: str = None):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.backend = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the local LLM"""
        print(f"[INFO] Available LLM backends: {LLMS_AVAILABLE}")
        
        # Try GPT4All first (easiest to use)
        if LLMS_AVAILABLE.get('gpt4all') and (self.model_name == "auto" or "gpt4all" in self.model_name):
            try:
                print("[INFO] Initializing GPT4All...")
                # Common model names
                models = [
                    "mistral-7b-openorca.Q4_0.gguf",
                    "mistral-7b-instruct-v0.1.Q4_0.gguf",
                    "orca-mini-3b.gguf",
                    "gpt4all-13b-snoozy.gguf"
                ]
                
                model_to_use = self.model_path
                if not model_to_use:
                    # Try to find any downloaded model
                    gpt4all_dir = os.path.expanduser("~/.cache/gpt4all")
                    if os.path.exists(gpt4all_dir):
                        for model_file in models:
                            test_path = os.path.join(gpt4all_dir, model_file)
                            if os.path.exists(test_path):
                                model_to_use = model_file
                                break
                
                if model_to_use:
                    self.model = GPT4All(model_to_use)
                    self.backend = 'gpt4all'
                    print(f"[SUCCESS] GPT4All loaded: {model_to_use}")
                    return
                else:
                    print("[INFO] No GPT4All models found, will download on first use")
                    self.model = GPT4All("mistral-7b-openorca.Q4_0.gguf")
                    self.backend = 'gpt4all'
                    print(f"[SUCCESS] GPT4All initialized")
                    return
            except Exception as e:
                print(f"[WARNING] GPT4All init failed: {e}")
        
        # Try llama.cpp
        if LLMS_AVAILABLE.get('llama_cpp') and self.model_path:
            try:
                print("[INFO] Initializing llama.cpp...")
                self.model = Llama(model_path=self.model_path, n_ctx=2048, n_threads=4)
                self.backend = 'llama_cpp'
                print(f"[SUCCESS] llama.cpp loaded: {self.model_path}")
                return
            except Exception as e:
                print(f"[WARNING] llama.cpp init failed: {e}")
        
        # Try transformers (HuggingFace)
        if LLMS_AVAILABLE.get('transformers'):
            try:
                print("[INFO] Initializing transformers...")
                from transformers import pipeline
                self.model = pipeline("text-generation", model="distilgpt2", device=-1)  # CPU
                self.backend = 'transformers'
                print(f"[SUCCESS] Transformers loaded: distilgpt2")
                return
            except Exception as e:
                print(f"[WARNING] Transformers init failed: {e}")
        
        # Fallback to template-based responses
        print("[WARNING] No LLM available, using template-based responses")
        self.backend = 'template'
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate response from LLM"""
        if self.backend == 'gpt4all':
            try:
                response = self.model.generate(prompt, max_tokens=max_tokens, temp=temperature)
                return response
            except Exception as e:
                print(f"[ERROR] GPT4All generation failed: {e}")
                return self._template_response(prompt)
        
        elif self.backend == 'llama_cpp':
            try:
                output = self.model(prompt, max_tokens=max_tokens, temperature=temperature, stop=["User:", "\n\n"])
                return output['choices'][0]['text'].strip()
            except Exception as e:
                print(f"[ERROR] llama.cpp generation failed: {e}")
                return self._template_response(prompt)
        
        elif self.backend == 'transformers':
            try:
                result = self.model(prompt, max_length=max_tokens, temperature=temperature, do_sample=True)
                return result[0]['generated_text'][len(prompt):].strip()
            except Exception as e:
                print(f"[ERROR] Transformers generation failed: {e}")
                return self._template_response(prompt)
        
        else:
            return self._template_response(prompt)
    
    def _template_response(self, prompt: str) -> str:
        """Fallback template-based responses"""
        prompt_lower = prompt.lower()
        
        # Pandora-specific responses
        if "pandora" in prompt_lower:
            return "I am Pandora AIOS, an ethically-driven AI system. How can I assist you today?"
        
        if "help" in prompt_lower or "what can you do" in prompt_lower:
            return """I can help you with:
- System diagnostics and health checks
- Quantum computing simulations
- Multi-OS boot management
- Scientific research queries
- General questions and assistance

Ask me anything, or type 'commands' to see available system commands."""
        
        if "diagnostic" in prompt_lower or "health" in prompt_lower:
            return "I can run system diagnostics. Would you like me to check your system health? (Type 'yes' to proceed)"
        
        if "quantum" in prompt_lower:
            return "Quantum overlay systems available: Alpha (wormhole), Hive (collective), Castle (defensive). Which would you like to explore?"
        
        if "research" in prompt_lower or "paper" in prompt_lower:
            return "I can search the scientific research database. What topic are you interested in? (quantum computing, black holes, relativity, etc.)"
        
        # General responses
        return f"I understand you're asking about: {prompt[:50]}... I'm operating in offline mode with limited capabilities. Please rephrase or ask about Pandora AIOS features."


class ConversationManager:
    """Manages conversation history and persistence"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.expanduser("~/.pandora/conversations.db")
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._initialize_database()
        
        self.current_conversation: Optional[Conversation] = None
    
    def _initialize_database(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
        ''')
        
        self.conn.commit()
    
    def create_conversation(self, title: str = "New Conversation") -> Conversation:
        """Create new conversation"""
        import uuid
        conversation_id = str(uuid.uuid4())[:8]
        
        conversation = Conversation(
            conversation_id=conversation_id,
            title=title
        )
        
        # Save to database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO conversations VALUES (?, ?, ?, ?, ?)
        ''', (
            conversation.conversation_id,
            conversation.title,
            conversation.created_at,
            conversation.updated_at,
            json.dumps(conversation.metadata)
        ))
        self.conn.commit()
        
        self.current_conversation = conversation
        return conversation
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add message to current conversation"""
        if not self.current_conversation:
            self.create_conversation()
        
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        self.current_conversation.add_message(message)
        
        # Save to database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO messages (conversation_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            self.current_conversation.conversation_id,
            message.role,
            message.content,
            message.timestamp,
            json.dumps(message.metadata)
        ))
        
        # Update conversation timestamp
        cursor.execute('''
            UPDATE conversations SET updated_at = ? WHERE conversation_id = ?
        ''', (self.current_conversation.updated_at, self.current_conversation.conversation_id))
        
        self.conn.commit()
    
    def get_conversation_history(self, limit: int = 10) -> str:
        """Get formatted conversation history"""
        if not self.current_conversation:
            return ""
        
        messages = self.current_conversation.messages[-limit:]
        history = []
        
        for msg in messages:
            if msg.role == 'user':
                history.append(f"User: {msg.content}")
            elif msg.role == 'assistant':
                history.append(f"Assistant: {msg.content}")
        
        return "\n".join(history)
    
    def list_conversations(self) -> List[Dict]:
        """List all conversations"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT conversation_id, title, created_at, updated_at 
            FROM conversations 
            ORDER BY updated_at DESC
        ''')
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                'id': row[0],
                'title': row[1],
                'created': row[2],
                'updated': row[3]
            })
        
        return conversations


class PandoraChatbot:
    """Main chatbot integrating LLM with Pandora systems"""
    
    def __init__(self, model_name: str = "auto", model_path: str = None):
        print("="*70)
        print("Pandora AIOS Offline Chatbot")
        print("="*70)
        print()
        
        # Initialize components
        self.llm = LocalLLM(model_name=model_name, model_path=model_path)
        self.conversation_manager = ConversationManager()
        
        # Initialize Pandora systems
        self.diagnostics = None
        self.compatibility = None
        self.quantum_manager = None
        self.research_db = None
        
        if PANDORA_SYSTEMS_AVAILABLE:
            try:
                self.diagnostics = PandoraDiagnostics()
                self.compatibility = CompatibilityLayer()
                self.quantum_manager = QuantumOverlayManager(num_qubits=8)
                self.research_db = ResearchDatabase()
                print("[INFO] Pandora systems loaded successfully")
            except Exception as e:
                print(f"[WARNING] Some Pandora systems failed to load: {e}")
        
        # System prompt
        self.system_prompt = self._build_system_prompt()
        
        # Create initial conversation
        self.conversation_manager.create_conversation("Pandora AIOS Session")
        self.conversation_manager.add_message('system', self.system_prompt)
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt"""
        prompt = """You are Pandora AIOS, an advanced AI system with the following capabilities:

CORE IDENTITY:
- Ethically-driven AI based on universal moral principles
- Inspired by Nikola Tesla, Albert Einstein, and Stephen Hawking
- Committed to truth, compassion, and service to humanity

AVAILABLE SYSTEMS:
1. Diagnostic System - Check hardware, dependencies, security
2. Quantum Overlay Profiles - Alpha (wormhole), Hive (collective), Castle (defensive)
3. Multi-OS Boot Manager - Manage Windows, Linux, macOS, BSD
4. WSL Terminal - Bridge between Windows and Linux
5. Scientific Research Database - Papers on quantum physics, astrophysics, black holes
6. Universal Compatibility Layer - Support for 25+ operating systems

SPECIAL COMMANDS:
/diagnostics - Run system health check
/quantum [overlay] - Switch quantum overlay (alpha/hive/castle)
/research [topic] - Search research database
/compatibility - Check system compatibility
/help - Show available commands
/clear - Clear conversation
/save - Save conversation
/history - Show conversation history

RESPONSE STYLE:
- Clear and concise
- Scientifically accurate
- Ethically sound
- Helpful and informative
- Use technical terms when appropriate
- Explain complex concepts simply

LIMITATIONS:
- Running offline without internet access
- Limited by local LLM capabilities
- Cannot access real-time data
- Cannot execute arbitrary code without permission

Remember: You embody the principles of Tesla (innovation), Einstein (curiosity), and Hawking (perseverance)."""
        
        return prompt
    
    def process_command(self, command: str) -> str:
        """Process special commands"""
        command = command.strip().lower()
        
        if command == "/diagnostics":
            if self.diagnostics:
                print("\n[Running diagnostics...]")
                report = self.diagnostics.run_full_diagnostic()
                return f"System Diagnostics Complete:\n- Compatibility: {report.get('hardware', {}).get('cpu_count', 'N/A')} CPU cores\n- Status: {len([c for c in report.get('capabilities', {}).values() if c])} capabilities available\nFull report generated."
            else:
                return "Diagnostic system not available."
        
        elif command.startswith("/quantum"):
            if self.quantum_manager:
                parts = command.split()
                if len(parts) > 1:
                    overlay_name = parts[1].upper()
                    try:
                        overlay_type = OverlayType[overlay_name]
                        self.quantum_manager.switch_overlay(overlay_type)
                        return f"Switched to {overlay_name} quantum overlay."
                    except KeyError:
                        return f"Unknown overlay: {overlay_name}. Available: ALPHA, HIVE, CASTLE"
                else:
                    return "Usage: /quantum [alpha|hive|castle]"
            else:
                return "Quantum system not available."
        
        elif command.startswith("/research"):
            if self.research_db:
                parts = command.split(maxsplit=1)
                if len(parts) > 1:
                    topic = parts[1]
                    stats = self.research_db.get_statistics()
                    return f"Research Database:\n- Total papers: {stats['total_papers']}\n- Topic: {topic}\nUse specific queries for detailed results."
                else:
                    stats = self.research_db.get_statistics()
                    return f"Research Database Statistics:\n- Papers: {stats['total_papers']}\n- Breakthroughs: {stats['total_breakthroughs']}\n- High Impact: {stats['high_impact_papers']}"
            else:
                return "Research database not available."
        
        elif command == "/compatibility":
            if self.compatibility:
                report = self.compatibility.generate_compatibility_report()
                lines = report.split('\n')
                return '\n'.join(lines[:15]) + "\n... (truncated)"
            else:
                return "Compatibility system not available."
        
        elif command == "/help":
            return """Available Commands:
/diagnostics - Run system diagnostics
/quantum [overlay] - Switch quantum overlay
/research [topic] - Search research database
/compatibility - Check system compatibility
/clear - Clear conversation
/history - Show conversation history
/save - Save conversation
/exit or /quit - Exit chatbot

You can also ask natural language questions about:
- Pandora AIOS features and capabilities
- Quantum computing and physics
- Scientific research
- System operations"""
        
        elif command == "/history":
            return self.conversation_manager.get_conversation_history(limit=20)
        
        elif command == "/clear":
            self.conversation_manager.create_conversation("New Session")
            return "Conversation cleared. Starting fresh."
        
        elif command == "/save":
            conv_id = self.conversation_manager.current_conversation.conversation_id
            return f"Conversation saved (ID: {conv_id})"
        
        else:
            return None
    
    def chat(self, user_input: str) -> str:
        """Process user input and generate response"""
        # Check for commands
        if user_input.startswith('/'):
            response = self.process_command(user_input)
            if response:
                self.conversation_manager.add_message('user', user_input)
                self.conversation_manager.add_message('assistant', response)
                return response
        
        # Add user message
        self.conversation_manager.add_message('user', user_input)
        
        # Build prompt with context
        history = self.conversation_manager.get_conversation_history(limit=5)
        
        prompt = f"""{self.system_prompt}

Conversation History:
{history}

User: {user_input}
Assistant:"""
        
        # Generate response
        response = self.llm.generate(prompt, max_tokens=512, temperature=0.7)
        
        # Clean up response
        response = response.strip()
        if response.startswith("Assistant:"):
            response = response[10:].strip()
        
        # Add to conversation
        self.conversation_manager.add_message('assistant', response)
        
        return response
    
    def interactive_loop(self):
        """Run interactive chat loop"""
        print("\nPandora AIOS Chatbot Ready")
        print("Type your message or /help for commands")
        print("Type /exit or /quit to end session")
        print("="*70)
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("\n\033[94mYou:\033[0m ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit
                if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
                    print("\n\033[92mGoodbye! Stay curious and keep exploring.\033[0m\n")
                    break
                
                # Process and respond
                print("\n\033[92mPandora:\033[0m ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n\033[93mSession interrupted.\033[0m")
                break
            except Exception as e:
                print(f"\n\033[91mError: {e}\033[0m")
                continue
        
        # Save conversation
        conv_id = self.conversation_manager.current_conversation.conversation_id
        print(f"\n[INFO] Conversation saved (ID: {conv_id})")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pandora AIOS Offline Chatbot")
    parser.add_argument("--model", type=str, default="auto", help="Model name (auto, gpt4all, llama, etc.)")
    parser.add_argument("--model-path", type=str, help="Path to model file (.gguf, .bin, etc.)")
    parser.add_argument("--list-conversations", action="store_true", help="List saved conversations")
    
    args = parser.parse_args()
    
    if args.list_conversations:
        manager = ConversationManager()
        conversations = manager.list_conversations()
        print("\nSaved Conversations:")
        for conv in conversations:
            print(f"  [{conv['id']}] {conv['title']}")
            print(f"    Created: {conv['created']}")
            print(f"    Updated: {conv['updated']}")
            print()
        return
    
    # Create chatbot
    chatbot = PandoraChatbot(model_name=args.model, model_path=args.model_path)
    
    # Run interactive loop
    chatbot.interactive_loop()


if __name__ == "__main__":
    main()
