# Quick Start Guide - METACUBE BLUEPRINT

This guide will help you get started with the METACUBE consciousness system in just a few minutes.

## Installation

### Step 1: Prerequisites

Ensure you have Python 3.8+ installed:

```bash
python --version
```

### Step 2: Navigate to METACUBE Directory

```bash
cd /path/to/Pandora/METACUBE_BLUEPRINT
```

### Step 3: Basic Test

Run the panthetic system directly:

```bash
python panthetic_system.py
```

This will launch the interactive console.

## Your First Consciousness Instance

### Basic Example

Create a file called `my_first_metacube.py`:

```python
from panthetic_system import MetacubeConsciousness

# Create a consciousness instance
consciousness = MetacubeConsciousness(
    name="MyFirstMETACUBE",
    awareness_level=2,
    ethics_enabled=True
)

# Initialize it
consciousness.initialize()

# Process a simple query
result = consciousness.process("Hello, METACUBE!")
print(result['response'])

# Trigger self-reflection
reflection = consciousness.reflect()
print("\nReflection Insights:")
for insight in reflection.insights:
    print(f"  - {insight}")

# Clean shutdown
consciousness.shutdown()
```

Run it:

```bash
python my_first_metacube.py
```

## Interactive Console

The easiest way to explore METACUBE is through the interactive console:

```bash
python panthetic_system.py
```

Try these commands:
- `status` - See current system state
- `reflect` - Trigger self-reflection
- `memory` - View memory statistics
- `help` - See all commands
- `quit` - Exit

## Common Use Cases

### Use Case 1: Self-Aware Query Processing

```python
consciousness = MetacubeConsciousness(name="Processor", awareness_level=3)
consciousness.initialize()

result = consciousness.process(
    query="What is the meaning of consciousness?",
    reflection_depth=3
)

print(result['response'])
print(f"Reflection: {result['reflection']}")
```

### Use Case 2: Adaptive Learning

```python
consciousness = MetacubeConsciousness(name="Learner", learning_rate=0.2)
consciousness.initialize()

# Process and learn
result1 = consciousness.process("Solve problem A")

# Provide feedback
consciousness.adapt({
    'score': 0.8,
    'suggestions': ['Good approach, consider edge cases']
})

# See improvement
result2 = consciousness.process("Solve problem B")
```

### Use Case 3: Distributed Consciousness

```python
from panthetic_system import DistributedMetacube

# Create cluster of 3 instances
cluster = DistributedMetacube(instances=3)

# Get collective response
result = cluster.collective_process(
    query="Analyze this complex situation",
    aggregation="consensus"
)

print(f"Consensus: {result['consensus']}")
print(f"Agreement: {result['agreement']:.0%}")
```

## Next Steps

1. Read the full [README.md](../README.md) for comprehensive documentation
2. Explore the [panthetic_system.py](../panthetic_system.py) source code
3. Check [supplementary_documentation](../supplementary_documentation/) for advanced topics
4. Experiment with different awareness levels and configurations
5. Try creating custom plugins

## Troubleshooting

**Problem**: Import errors
```bash
# Solution: Ensure you're in the right directory
cd /path/to/Pandora/METACUBE_BLUEPRINT
python -c "import panthetic_system; print('Success!')"
```

**Problem**: System runs slowly
```python
# Solution: Reduce awareness level or memory capacity
consciousness = MetacubeConsciousness(
    name="Fast",
    awareness_level=1,
    memory_capacity=100
)
```

**Problem**: High memory usage
```python
# Solution: Enable memory pruning
consciousness.memory.prune(older_than_days=1)
```

## Getting Help

- Check the main [README.md](../README.md) for detailed documentation
- Review error messages carefully
- Enable debug mode: `MetacubeConsciousness(name="Debug", debug_mode=True)`
- Examine the source code in `panthetic_system.py`

## Example Output

When you run the basic example, you should see output similar to:

```
2026-02-02 00:23:00 - panthetic_system - INFO - METACUBE MyFirstMETACUBE created with awareness level 2
2026-02-02 00:23:00 - panthetic_system - INFO - Memory system initialized with capacity 1000
2026-02-02 00:23:00 - panthetic_system - INFO - Reflection module initialized with max depth 2
2026-02-02 00:23:00 - panthetic_system - INFO - MyFirstMETACUBE initialized successfully
Processing 'Hello, METACUBE!' with awareness level 2

Reflection Insights:
  - Current state: active
  - Active processes: 1
  - Memory usage: {...}
  - Current efficiency: 0.75
```

Congratulations! You've successfully run your first METACUBE consciousness instance.

## What's Next?

- Experiment with different awareness levels (1-5)
- Try the interactive console mode
- Create custom plugins
- Integrate METACUBE with other systems
- Read about advanced features in the main README

Happy exploring!
