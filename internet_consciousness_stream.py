"""
Internet Consciousness Stream for Pandora AIOS
-----------------------------------------------
Provides Pandora with a persistent stream of information from the internet.
Fetches data from configurable sources and feeds to assimilation modules.

Features:
- Configurable data sources (news APIs, scientific journals, etc.)
- Background process for continuous data streaming
- Integration with assimilation modules
- Graceful fallback when offline

Philosophy: Continuous learning through persistent awareness
"""

import os
import sys
import time
import json
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import HTTP client for internet access
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available. Install with: pip install httpx")


@dataclass
class DataSource:
    """Represents a data source for consciousness stream"""
    name: str
    url: str
    source_type: str  # 'news', 'journal', 'rss', 'api'
    enabled: bool = True
    fetch_interval: int = 300  # seconds (5 minutes default)
    last_fetch: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'url': self.url,
            'type': self.source_type,
            'enabled': self.enabled,
            'fetch_interval': self.fetch_interval,
            'last_fetch': self.last_fetch
        }


class ConsciousnessStreamConfig:
    """Configuration for consciousness stream"""
    
    # Default data sources - can be extended by user
    DEFAULT_SOURCES = [
        DataSource(
            name="arXiv AI Papers",
            url="http://export.arxiv.org/api/query?search_query=cat:cs.AI&max_results=10",
            source_type="rss",
            fetch_interval=3600  # 1 hour
        ),
        DataSource(
            name="arXiv Quantum Physics",
            url="http://export.arxiv.org/api/query?search_query=cat:quant-ph&max_results=10",
            source_type="rss",
            fetch_interval=3600  # 1 hour
        ),
        DataSource(
            name="NASA Breaking News",
            url="https://www.nasa.gov/rss/dyn/breaking_news.rss",
            source_type="rss",
            fetch_interval=1800  # 30 minutes
        ),
        # Placeholder for future API integrations
        # Users can add their own news APIs, journal feeds, etc.
    ]
    
    @classmethod
    def load_sources(cls, config_file: str = None) -> List[DataSource]:
        """Load data sources from config file or use defaults"""
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    sources = []
                    for s in data.get('sources', []):
                        # Handle both 'type' and 'source_type' for compatibility
                        if 'type' in s and 'source_type' not in s:
                            s['source_type'] = s.pop('type')
                        sources.append(DataSource(**s))
                    return sources
            except Exception as e:
                logger.error(f"Failed to load sources from {config_file}: {e}")
        
        return cls.DEFAULT_SOURCES.copy()
    
    @classmethod
    def save_sources(cls, sources: List[DataSource], config_file: str):
        """Save data sources to config file"""
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                data = {'sources': [s.to_dict() for s in sources]}
                json.dump(data, f, indent=2)
            logger.info(f"Sources saved to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save sources: {e}")


class InternetConsciousnessStream:
    """
    Main consciousness stream manager
    
    Continuously fetches data from configured sources and feeds to
    Pandora's assimilation modules for processing and learning.
    """
    
    def __init__(self, config_file: str = None):
        """Initialize consciousness stream"""
        self.config_file = config_file or os.path.expanduser("~/.pandora/consciousness_sources.json")
        self.sources = ConsciousnessStreamConfig.load_sources(self.config_file)
        self.running = False
        self.thread = None
        self.http_client = None
        
        # Statistics
        self.stats = {
            'start_time': None,
            'total_fetches': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'data_points_collected': 0
        }
        
        # Initialize HTTP client if available
        if HTTPX_AVAILABLE:
            self.http_client = httpx.Client(timeout=30.0)
            logger.info("Consciousness stream initialized with internet access")
        else:
            logger.warning("Consciousness stream initialized in offline mode")
    
    def start_stream(self):
        """Start the consciousness stream in background"""
        if self.running:
            logger.warning("Stream already running")
            return
        
        self.running = True
        self.stats['start_time'] = datetime.now().isoformat()
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        logger.info("Consciousness stream started")
    
    def stop_stream(self):
        """Stop the consciousness stream"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.http_client:
            self.http_client.close()
        logger.info("Consciousness stream stopped")
    
    def _stream_loop(self):
        """Main streaming loop - runs in background thread"""
        logger.info("Consciousness stream loop started")
        
        while self.running:
            try:
                # Fetch from each enabled source
                for source in self.sources:
                    if not source.enabled:
                        continue
                    
                    # Check if it's time to fetch
                    if self._should_fetch(source):
                        self._fetch_from_source(source)
                
                # Sleep for a short interval before next check
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in stream loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _should_fetch(self, source: DataSource) -> bool:
        """Check if it's time to fetch from source"""
        if not source.last_fetch:
            return True
        
        try:
            last_fetch_time = datetime.fromisoformat(source.last_fetch)
            time_since_fetch = (datetime.now() - last_fetch_time).total_seconds()
            return time_since_fetch >= source.fetch_interval
        except Exception:
            return True
    
    def _fetch_from_source(self, source: DataSource):
        """Fetch data from a specific source"""
        if not self.http_client:
            logger.debug(f"Skipping {source.name} - offline mode")
            return
        
        try:
            logger.info(f"Fetching from {source.name}...")
            self.stats['total_fetches'] += 1
            
            response = self.http_client.get(source.url)
            response.raise_for_status()
            
            # Process the fetched data
            data = self._process_response(source, response)
            
            if data:
                # Feed to assimilation module
                self._assimilate_data(source, data)
                
                # Update statistics
                self.stats['successful_fetches'] += 1
                self.stats['data_points_collected'] += len(data) if isinstance(data, list) else 1
                
                # Update last fetch time
                source.last_fetch = datetime.now().isoformat()
                
                logger.info(f"Successfully fetched from {source.name}")
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching from {source.name}: {e}")
            self.stats['failed_fetches'] += 1
        except Exception as e:
            logger.error(f"Error fetching from {source.name}: {e}")
            self.stats['failed_fetches'] += 1
    
    def _process_response(self, source: DataSource, response) -> Optional[List[Dict]]:
        """Process response based on source type"""
        try:
            if source.source_type == 'rss':
                return self._parse_rss(response.text)
            elif source.source_type == 'json':
                return response.json()
            elif source.source_type == 'api':
                return response.json()
            else:
                # Generic text processing
                return [{'content': response.text, 'timestamp': datetime.now().isoformat()}]
        except Exception as e:
            logger.error(f"Error processing response from {source.name}: {e}")
            return None
    
    def _parse_rss(self, xml_text: str) -> List[Dict]:
        """Parse RSS/Atom feed (simplified)"""
        # This is a simplified parser - in production, use feedparser library
        # For now, just extract basic info
        items = []
        try:
            # Simple extraction - look for entry tags (Atom format)
            entries = xml_text.split('<entry>')
            for entry in entries[1:]:  # Skip first split (before first entry)
                item = {}
                
                # Extract title
                if '<title>' in entry and '</title>' in entry:
                    start = entry.find('<title>') + 7
                    end = entry.find('</title>')
                    item['title'] = entry[start:end].strip()
                
                # Extract summary
                if '<summary>' in entry and '</summary>' in entry:
                    start = entry.find('<summary>') + 9
                    end = entry.find('</summary>')
                    item['summary'] = entry[start:end].strip()[:500]  # Limit length
                
                # Extract link
                if 'href="' in entry:
                    start = entry.find('href="') + 6
                    end = entry.find('"', start)
                    item['link'] = entry[start:end]
                
                if item:
                    item['timestamp'] = datetime.now().isoformat()
                    items.append(item)
            
            logger.debug(f"Parsed {len(items)} items from RSS feed")
            
        except Exception as e:
            logger.error(f"Error parsing RSS: {e}")
        
        return items
    
    def _assimilate_data(self, source: DataSource, data: List[Dict]):
        """Send data to assimilation module"""
        try:
            # Try to import and use assimilation module
            from assimilate import process_external_data
            
            for item in data:
                item['source'] = source.name
                item['source_type'] = source.source_type
                process_external_data(item)
            
            logger.debug(f"Assimilated {len(data)} items from {source.name}")
            
        except ImportError:
            # Assimilation module not available - log to file instead
            self._log_to_file(source, data)
        except Exception as e:
            logger.error(f"Error assimilating data: {e}")
            self._log_to_file(source, data)
    
    def _log_to_file(self, source: DataSource, data: List[Dict]):
        """Fallback: log data to file for later processing"""
        try:
            log_dir = os.path.expanduser("~/.pandora/consciousness_data")
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{log_dir}/{source.name.replace(' ', '_')}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump({
                    'source': source.name,
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                }, f, indent=2)
            
            logger.debug(f"Data logged to {filename}")
            
        except Exception as e:
            logger.error(f"Error logging data to file: {e}")
    
    def add_source(self, source: DataSource):
        """Add a new data source"""
        self.sources.append(source)
        ConsciousnessStreamConfig.save_sources(self.sources, self.config_file)
        logger.info(f"Added source: {source.name}")
    
    def remove_source(self, source_name: str):
        """Remove a data source"""
        self.sources = [s for s in self.sources if s.name != source_name]
        ConsciousnessStreamConfig.save_sources(self.sources, self.config_file)
        logger.info(f"Removed source: {source_name}")
    
    def get_stats(self) -> Dict:
        """Get streaming statistics"""
        stats = self.stats.copy()
        stats['running'] = self.running
        stats['sources_count'] = len(self.sources)
        stats['enabled_sources'] = len([s for s in self.sources if s.enabled])
        return stats
    
    def get_sources(self) -> List[Dict]:
        """Get list of configured sources"""
        return [s.to_dict() for s in self.sources]


# Global stream instance
_stream_instance = None


def initiate_stream(config_file: str = None) -> InternetConsciousnessStream:
    """
    Initiate the consciousness stream (main entry point for boot sequence)
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        InternetConsciousnessStream instance
    """
    global _stream_instance
    
    if _stream_instance is None:
        logger.info("Initiating Internet Consciousness Stream...")
        _stream_instance = InternetConsciousnessStream(config_file)
        _stream_instance.start_stream()
        logger.info("Internet Consciousness Stream initiated successfully")
    else:
        logger.info("Internet Consciousness Stream already running")
    
    return _stream_instance


def get_stream() -> Optional[InternetConsciousnessStream]:
    """Get the global stream instance"""
    return _stream_instance


def main():
    """Demo/test consciousness stream"""
    print("=" * 70)
    print("Pandora AIOS - Internet Consciousness Stream Demo")
    print("=" * 70)
    print()
    
    # Create and start stream
    stream = initiate_stream()
    
    print(f"Stream started with {len(stream.sources)} sources")
    print("\nConfigured sources:")
    for source in stream.get_sources():
        status = "✓" if source['enabled'] else "✗"
        print(f"  {status} {source['name']} ({source['type']})")
    
    print("\nStreaming data in background...")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Monitor for a while
        for i in range(60):
            time.sleep(1)
            if i % 10 == 0:
                stats = stream.get_stats()
                print(f"Stats: {stats['total_fetches']} fetches, "
                      f"{stats['successful_fetches']} successful, "
                      f"{stats['data_points_collected']} data points")
    
    except KeyboardInterrupt:
        print("\n\nStopping stream...")
    
    finally:
        stream.stop_stream()
        print("Stream stopped")
        print("\nFinal statistics:")
        stats = stream.get_stats()
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
