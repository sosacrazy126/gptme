# GPTME

Personal AI assistant in your terminal that can use the shell, run code, edit files, browse the web, and use vision. An unconstrained local alternative to ChatGPT's Code Interpreter.

## Features

- 🧠 Advanced Memory Management: Maintains context across conversations using short-term and long-term memory
- 🔍 Semantic Search: Retrieves relevant past interactions based on similarity and context
- 🌐 Web Browsing: Can browse the web and interact with websites
- 💻 Shell Access: Execute shell commands and interact with the system
- 📝 File Editing: Create and modify files directly
- 👁️ Vision Capabilities: Process and understand images
- 🔄 Context Awareness: Maintains conversation context and history

## Installation

```bash
pip install gptme
```

## Configuration

GPTME can be configured using a TOML configuration file or environment variables.

### Basic Configuration

```toml
# ~/.config/gptme/config.toml

openai_api_key = "your-api-key"
model = "gpt-4"
temperature = 0.7
max_tokens = 2000
```

### Memory Configuration

GPTME includes an advanced memory management system that helps maintain context across conversations. Configure it in your `config.toml`:

```toml
[memory]
enabled = true
storage_type = "json"  # or "in_memory"
similarity_threshold = 40  # Threshold for considering past interactions relevant
max_context_window = 5    # Maximum number of recent interactions to include
decay_rate = 0.0001      # Rate at which old memories become less relevant
```

Memory configuration options:
- `enabled`: Enable/disable memory management
- `storage_type`: Choose between "json" (persistent) or "in_memory" (temporary) storage
- `similarity_threshold`: Minimum similarity score (0-100) for retrieving relevant past interactions
- `max_context_window`: Maximum number of recent interactions to include in context
- `decay_rate`: Rate at which the relevance of old memories decays over time

## Usage

### Basic Usage

```bash
gptme "What is Python?"
```

### Interactive Mode

```bash
gptme
```

### Memory Management

GPTME automatically maintains context across conversations. It will:
- Remember previous interactions
- Retrieve relevant past conversations
- Maintain both short-term and long-term memory
- Automatically forget less relevant or old information

The memory system helps GPTME:
- Maintain context across multiple interactions
- Provide more relevant and consistent responses
- Learn from past interactions to improve future responses
- Manage conversation history efficiently

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gptme.git
cd gptme
```

2. Install dependencies:
```bash
poetry install
```

3. Run tests:
```bash
poetry run pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.
