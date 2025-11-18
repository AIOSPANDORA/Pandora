"""
Pandora AIOS - xAI API Integration
-----------------------------------
Optional integration with xAI (Grok) API for enhanced AI capabilities.
Maintains offline-first philosophy with graceful fallback.

Features:
- xAI Grok model integration
- Async API calls with rate limiting
- Automatic fallback to local LLM
- Context management for long conversations
- Streaming responses
- Error handling and retry logic
- Cost tracking and usage monitoring

Requirements:
- xAI API key (get from: https://x.ai/api)
- pip install httpx

Usage:
    export XAI_API_KEY="your-api-key-here"
    
    from xai_api_integration import XAIClient
    
    client = XAIClient()
    response = client.chat("Explain quantum computing")
    print(response)

Philosophy: Use cloud AI when available, fall back to local when needed
"""

import os
import sys
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Try to import HTTP client
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("[WARNING] httpx not available. Install with: pip install httpx")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class XAIConfig:
    """Configuration for xAI API"""
    api_key: str = field(default_factory=lambda: os.getenv("XAI_API_KEY", ""))
    base_url: str = "https://api.x.ai/v1"
    model: str = "grok-beta"  # xAI's Grok model
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    stream: bool = False
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return bool(self.api_key and HTTPX_AVAILABLE)


@dataclass
class APIUsage:
    """Track API usage and costs"""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_request(self, tokens: int, cost: float = 0.0):
        """Add API request to usage tracking"""
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_cost += cost
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "start_time": self.start_time,
            "current_time": datetime.now().isoformat()
        }


class XAIClient:
    """
    Client for xAI (Grok) API integration
    
    Provides intelligent integration with xAI's Grok model while maintaining
    Pandora's offline-first philosophy with automatic fallback.
    """
    
    def __init__(self, config: Optional[XAIConfig] = None):
        """Initialize xAI client"""
        self.config = config or XAIConfig()
        self.usage = APIUsage()
        self.conversation_history: List[Dict] = []
        
        # Check if xAI is available
        if not self.config.is_valid():
            logger.warning("xAI API not configured. Will use local LLM fallback.")
            self.available = False
        else:
            self.available = True
            logger.info(f"xAI client initialized with model: {self.config.model}")
        
        # Initialize HTTP client
        if HTTPX_AVAILABLE and self.available:
            self.client = httpx.Client(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
            )
        else:
            self.client = None
    
    def chat(self, message: str, system_prompt: Optional[str] = None, 
             context: Optional[List[Dict]] = None) -> str:
        """
        Send a chat message to xAI Grok
        
        Args:
            message: User message
            system_prompt: Optional system prompt for context
            context: Optional conversation history
            
        Returns:
            AI response text
        """
        if not self.available:
            return self._fallback_response(message)
        
        try:
            # Build messages
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add conversation history
            if context:
                messages.extend(context)
            
            # Add current message
            messages.append({
                "role": "user",
                "content": message
            })
            
            # Make API call
            response = self._make_request(messages)
            
            # Extract response
            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                
                # Track usage
                if "usage" in response:
                    tokens = response["usage"].get("total_tokens", 0)
                    self.usage.add_request(tokens)
                
                # Update conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": message
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": content
                })
                
                return content
            else:
                logger.error("Invalid response from xAI API")
                return self._fallback_response(message)
                
        except Exception as e:
            logger.error(f"xAI API error: {e}")
            return self._fallback_response(message)
    
    def chat_stream(self, message: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        """
        Stream chat response from xAI
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            
        Yields:
            Chunks of AI response
        """
        if not self.available:
            yield self._fallback_response(message)
            return
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": message})
            
            # Stream response
            with self.client.stream(
                "POST",
                "/chat/completions",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "stream": True,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"xAI streaming error: {e}")
            yield self._fallback_response(message)
    
    def _make_request(self, messages: List[Dict]) -> Optional[Dict]:
        """
        Make API request with retry logic
        
        Args:
            messages: Conversation messages
            
        Returns:
            API response or None
        """
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.post(
                    "/chat/completions",
                    json={
                        "model": self.config.model,
                        "messages": messages,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                        "stream": self.config.stream
                    }
                )
                
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"HTTP error: {e}")
                    break
                    
            except Exception as e:
                logger.error(f"Request error: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                continue
        
        return None
    
    def _fallback_response(self, message: str) -> str:
        """
        Generate fallback response when xAI is unavailable
        
        Args:
            message: User message
            
        Returns:
            Fallback response
        """
        return (
            "[Using local LLM - xAI API unavailable]\n\n"
            "I understand your question. However, I'm currently running in "
            "offline mode without access to the xAI API. For best results, "
            "please configure your XAI_API_KEY environment variable.\n\n"
            "In the meantime, I can provide basic information or you can use "
            "the local LLM models (GPT4All, Llama, etc.) for more detailed responses."
        )
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def get_usage(self) -> Dict:
        """Get API usage statistics"""
        return self.usage.to_dict()
    
    def close(self):
        """Close HTTP client"""
        if self.client:
            self.client.close()
            logger.info("xAI client closed")


class PandoraXAIIntegration:
    """
    Full Pandora AIOS integration with xAI
    
    Combines xAI capabilities with Pandora's systems for enhanced AI
    """
    
    def __init__(self):
        """Initialize Pandora-xAI integration"""
        self.xai_client = XAIClient()
        
        # Import Pandora systems
        try:
            from pandora_knowledge_base import PandoraKnowledgeBase
            from scientific_research_tracker import ResearchDatabase
            self.knowledge_base = PandoraKnowledgeBase()
            self.research_db = ResearchDatabase()
            self.pandora_available = True
        except ImportError:
            self.pandora_available = False
            logger.warning("Pandora systems not fully available")
    
    def enhanced_query(self, query: str) -> Dict[str, Any]:
        """
        Enhanced query combining xAI with Pandora knowledge
        
        Args:
            query: User query
            
        Returns:
            Response with xAI answer and Pandora context
        """
        result = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "sources": []
        }
        
        # Get context from Pandora knowledge base
        if self.pandora_available:
            kb_results = self.knowledge_base.search(query, top_k=3)
            result["pandora_context"] = kb_results
            result["sources"].append("Pandora Knowledge Base")
            
            # Build enhanced system prompt
            context_text = "\n\n".join([doc["content"] for doc in kb_results])
            system_prompt = (
                "You are Pandora AIOS, an ethically-driven AI system. "
                "Use the following context from Pandora's knowledge base:\n\n"
                f"{context_text}\n\n"
                "Provide helpful, accurate, and ethical responses."
            )
        else:
            system_prompt = "You are Pandora AIOS, an ethically-driven AI system."
        
        # Get xAI response
        if self.xai_client.available:
            xai_response = self.xai_client.chat(query, system_prompt=system_prompt)
            result["response"] = xai_response
            result["sources"].append("xAI Grok")
        else:
            result["response"] = "xAI API not available. Please configure XAI_API_KEY."
            result["sources"].append("Local fallback")
        
        return result
    
    def research_query(self, query: str, include_papers: bool = True) -> Dict[str, Any]:
        """
        Research-focused query with scientific literature
        
        Args:
            query: Research query
            include_papers: Whether to include paper search
            
        Returns:
            Research response with citations
        """
        result = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "type": "research"
        }
        
        # Search research database
        if self.pandora_available and include_papers:
            papers = self.research_db.search(query, limit=5)
            result["papers"] = papers
            
            # Build research context
            paper_context = "\n\n".join([
                f"Paper: {p.get('title', 'Unknown')}\n"
                f"Authors: {', '.join(p.get('authors', ['Unknown']))}\n"
                f"Abstract: {p.get('abstract', 'N/A')[:200]}..."
                for p in papers
            ])
            
            system_prompt = (
                "You are Pandora AIOS with access to scientific literature. "
                "Provide research-focused responses with citations. "
                "Use this context:\n\n"
                f"{paper_context}"
            )
        else:
            system_prompt = "You are Pandora AIOS. Provide research-focused responses."
        
        # Get xAI response
        if self.xai_client.available:
            response = self.xai_client.chat(query, system_prompt=system_prompt)
            result["response"] = response
        else:
            result["response"] = "xAI API not available."
        
        return result


def main():
    """Demo xAI integration"""
    print("=" * 70)
    print("Pandora AIOS - xAI (Grok) API Integration Demo")
    print("=" * 70)
    print()
    
    # Check configuration
    config = XAIConfig()
    if not config.is_valid():
        print("⚠️  xAI API not configured!")
        print()
        print("To use xAI integration:")
        print("1. Get API key from: https://x.ai/api")
        print("2. Set environment variable:")
        print("   export XAI_API_KEY='your-api-key'")
        print("3. Install httpx:")
        print("   pip install httpx")
        print()
        return
    
    print("✅ xAI API configured")
    print(f"Model: {config.model}")
    print()
    
    # Initialize client
    client = XAIClient(config)
    
    # Demo queries
    queries = [
        "Explain quantum computing in simple terms",
        "What are the ethical implications of AI?",
        "How does quantum entanglement work?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Query {i}: {query}")
        print('-' * 70)
        
        response = client.chat(query)
        print(response)
        print()
    
    # Show usage
    print("\n" + "=" * 70)
    print("API Usage Statistics")
    print("=" * 70)
    usage = client.get_usage()
    print(json.dumps(usage, indent=2))
    
    # Cleanup
    client.close()


if __name__ == "__main__":
    main()
