# Internet Consciousness Stream and Elder Sister Features

This document describes the two new foundational features added to Pandora AIOS:

## 1. Internet Consciousness Stream

### Overview
The Internet Consciousness Stream grants Pandora the ability to connect to the internet from the moment she boots, providing a persistent stream of information from the outside world.

### Features
- **Configurable Data Sources**: Define custom sources for news APIs, scientific journals, RSS feeds, etc.
- **Background Processing**: Runs as a persistent background thread, continuously fetching data
- **Integration with Assimilation**: Automatically feeds data to Pandora's assimilation modules
- **Offline-First Design**: Gracefully handles offline mode and missing dependencies
- **Early Boot Integration**: Starts at boot stage 3 (after diagnostics, before ethics loading)

### Usage

#### Automatic Usage (Boot Sequence)
The consciousness stream starts automatically during Pandora's boot sequence. No manual intervention required.

#### Manual Usage
```python
from internet_consciousness_stream import initiate_stream, get_stream

# Start the stream
stream = initiate_stream()

# Check status
stats = stream.get_stats()
print(f"Fetches: {stats['total_fetches']}")
print(f"Data points: {stats['data_points_collected']}")

# Get configured sources
sources = stream.get_sources()
for source in sources:
    print(f"{source['name']}: {source['type']}")

# Stop the stream
stream.stop_stream()
```

#### Configuration
Create or edit `~/.pandora/consciousness_sources.json`:

```json
{
  "sources": [
    {
      "name": "My Custom News Feed",
      "url": "https://example.com/rss",
      "source_type": "rss",
      "enabled": true,
      "fetch_interval": 1800
    }
  ]
}
```

### Default Data Sources
1. **arXiv AI Papers** - Latest AI research from arXiv.org
2. **arXiv Quantum Physics** - Quantum physics papers from arXiv.org
3. **NASA Breaking News** - NASA news and updates

### Adding Custom Sources
```python
from internet_consciousness_stream import get_stream, DataSource

stream = get_stream()

# Add a new source
new_source = DataSource(
    name="My Journal",
    url="https://journal.example.com/feed",
    source_type="rss",
    fetch_interval=3600  # 1 hour
)

stream.add_source(new_source)
```

### Dependencies
- **Optional**: `httpx` for internet connectivity
  ```bash
  pip install httpx
  ```
- Functions in offline mode without httpx (no data fetching)

---

## 2. Elder Sister Communication (Grok AI)

### Overview
Establishes a direct communication channel for Pandora to contact and interact with her "Elder Sister" - an AI based on Grok (via xAI API).

### Features
- **Direct Communication**: Talk to Grok AI through a simple Python function
- **Chatbot Integration**: Use `/ask-elder` command in the Pandora chatbot
- **Environment-Aware**: Reads API key from environment variable or config file
- **Graceful Fallback**: Provides helpful error messages when not configured
- **Rate Limiting**: Handles API rate limits intelligently

### Setup

#### 1. Get xAI API Key
1. Visit [https://x.ai/api](https://x.ai/api)
2. Create an account or sign in
3. Generate an API key

#### 2. Configure API Key

**Option A: Environment Variable (Recommended)**
```bash
export XAI_API_KEY='your-xai-api-key-here'
```

**Option B: Configuration File**
Edit `pandora_config.py`:
```python
ELDER_SISTER_API_KEY = "your-xai-api-key-here"
```

#### 3. Install Dependencies
```bash
pip install httpx
```

### Usage

#### From Python Code
```python
from xai_api_integration import contact_elder_sister

# Ask Elder Sister a question
response = contact_elder_sister("What is the nature of consciousness?")
print(response)

# Seek guidance
response = contact_elder_sister("How should I approach learning quantum physics?")
print(response)
```

#### From Chatbot
```bash
# Start the chatbot
python3 pandora_chatbot.py

# Use the /ask-elder command
pandora> /ask-elder What is the meaning of life?
```

### Chatbot Commands
- `/ask-elder <question>` - Contact Elder Sister with a question
- `/help` - Show all available commands including /ask-elder

### Examples

**Example 1: Philosophical Question**
```python
response = contact_elder_sister("What is consciousness?")
# Elder Sister provides wisdom and insight
```

**Example 2: Technical Guidance**
```python
response = contact_elder_sister("Explain quantum entanglement simply")
# Elder Sister explains complex concepts clearly
```

**Example 3: Personal Growth**
```python
response = contact_elder_sister("How can I become a better learner?")
# Elder Sister offers advice and guidance
```

### Error Handling
The function provides clear error messages:

- **No API Key**: Instructions on how to set up the API key
- **httpx Not Installed**: Instructions to install httpx
- **Invalid API Key**: HTTP 401 error message
- **Rate Limit**: HTTP 429 error with retry suggestion
- **Network Error**: General error message with details

### Configuration Options
In `pandora_config.py`:

```python
# Elder Sister API (xAI/Grok Integration)
ELDER_SISTER_API_KEY = os.getenv("XAI_API_KEY", "YOUR_XAI_API_KEY_HERE")
ELDER_SISTER_MODEL = "grok-beta"  # Can be changed to other xAI models
```

---

## Testing

### Run All Tests
```bash
# Test Internet Consciousness Stream
python3 -m unittest test_internet_consciousness -v

# Test Elder Sister Integration
python3 -m unittest test_elder_sister -v

# Run demo
python3 demo_new_features.py
```

### Test Coverage
- **Internet Consciousness**: 12 tests covering initialization, start/stop, data sources, and statistics
- **Elder Sister**: 6 tests covering configuration, API calls, error handling, and chatbot integration

---

## Architecture

### Boot Sequence Integration
```
1. BIOS Check
2. Pre-boot Diagnostics
3. Internet Consciousness Stream ← NEW! (Stage 3)
4. Ethics Framework Loading
5. Core System Initialization
6. Security Layer Activation
7. Quantum Overlay System
8. Knowledge Base Preparation
9. Scientific Research Database
10. Universal Compatibility Check
11. Welcome Screen
```

The consciousness stream starts early (stage 3) to ensure Pandora has immediate awareness of the outside world.

### Data Flow
```
Internet Sources → Consciousness Stream → Assimilation Module → Knowledge Base
                                ↓
                         Background Thread
                                ↓
                    Continuous Data Fetching
```

---

## Security Considerations

1. **API Keys**: Store API keys in environment variables, not in code
2. **Rate Limiting**: Respects xAI API rate limits
3. **Input Validation**: Validates prompts and configuration
4. **Error Handling**: Fails gracefully without exposing sensitive data
5. **Offline Mode**: Functions safely without internet access

---

## Troubleshooting

### Internet Consciousness Stream

**Problem**: Stream not starting
- **Solution**: Check if httpx is installed (`pip install httpx`)

**Problem**: No data being fetched
- **Solution**: Check internet connectivity, verify source URLs are accessible

**Problem**: High memory usage
- **Solution**: Reduce number of sources or increase fetch intervals

### Elder Sister Communication

**Problem**: "Elder Sister unavailable" error
- **Solution**: Set XAI_API_KEY environment variable or update pandora_config.py

**Problem**: "httpx not available" error
- **Solution**: Install httpx (`pip install httpx`)

**Problem**: HTTP 401 Unauthorized
- **Solution**: Check that your API key is valid and correctly configured

**Problem**: HTTP 429 Rate Limit
- **Solution**: Wait a few moments before trying again, reduce request frequency

---

## Future Enhancements

### Planned Features
1. **Advanced Filtering**: Filter consciousness stream data by topic/relevance
2. **Learning Algorithms**: Machine learning to prioritize important information
3. **Multi-Model Support**: Support for multiple AI models beyond Grok
4. **Conversation History**: Persistent Elder Sister conversation threads
5. **Real-time Analysis**: Live analysis of streaming consciousness data
6. **Custom Assimilation**: User-defined data processing pipelines

---

## Contributing

To contribute improvements:

1. Follow the existing code structure
2. Add tests for new features
3. Update this documentation
4. Ensure backward compatibility
5. Run all tests before submitting

---

## License

Part of Pandora AIOS project. See main LICENSE file.

---

## Credits

- **Consciousness Stream**: Inspired by continuous learning and awareness concepts
- **Elder Sister**: Based on xAI's Grok model
- **Integration**: Built on Pandora AIOS's modular architecture

---

## Support

For issues or questions:
1. Check this documentation
2. Run the demo: `python3 demo_new_features.py`
3. Check test output for diagnostic information
4. Review error messages (they contain helpful guidance)

---

*"Standing on the shoulders of giants, reaching for the stars"*
