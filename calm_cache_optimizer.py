"""
CalmCacheOptimizer
Adaptive, stoic cache that smooths request spikes and prevents system overload.
Philosophy: Calm, moderation, never overreacts, preserves system harmony.
"""
import time
from collections import OrderedDict

class CalmCacheOptimizer:
    def __init__(self, max_entries=128, max_age_sec=180):
        self.cache = OrderedDict()
        self.max_entries = max_entries
        self.max_age_sec = max_age_sec
        self.hits = 0
        self.misses = 0

    def get(self, key):
        """Retrieve value if not expired"""
        if key not in self.cache:
            self.misses += 1
            return None
            
        value, timestamp = self.cache[key]
        age = time.time() - timestamp
        
        if age > self.max_age_sec:
            # Calmly remove expired entry
            del self.cache[key]
            self.misses += 1
            return None
        
        # Move to end (LRU ordering)
        self.cache.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key, value):
        """Gently trim old entries, add new value"""
        # Remove existing key if present
        if key in self.cache:
            del self.cache[key]
        
        # Gently trim if at capacity
        if len(self.cache) >= self.max_entries:
            # Remove oldest entry (first in OrderedDict)
            self.cache.popitem(last=False)
        
        # Add new entry with timestamp
        self.cache[key] = (value, time.time())

    def maintenance(self):
        """Remove expired entries calmly, without harsh sweeps"""
        current_time = time.time()
        expired_keys = []
        
        # Identify expired keys
        for key, (value, timestamp) in self.cache.items():
            if current_time - timestamp > self.max_age_sec:
                expired_keys.append(key)
        
        # Gently remove expired entries
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def stats(self):
        """Return cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'size': len(self.cache),
            'max_entries': self.max_entries,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%"
        }
    
    def clear(self):
        """Clear all cache entries (emergency use only)"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
class CalmCacheOptimizer:
    def __init__(self, max_entries=128, max_age_sec=180):
        self.cache = {}
        self.max_entries = max_entries
        self.max_age_sec = max_age_sec

    def get(self, key):
        # ... retrieve if not expired ...
        pass

    def set(self, key, value):
        # ... gently trim old entries, add new value ...
        pass

    def maintenance(self):
        # Remove expired calmly, without harsh sweeps
        pass
