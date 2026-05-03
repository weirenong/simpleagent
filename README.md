# SimpleAgent TUI

A lightweight terminal interface for SimpleAgent powered by Ollama.

## Features

- Chat with Ollama models in a terminal interface
- Command suggestions with slash commands
- Streaming responses with thinking display
- Memory management for conversation context
- Model and embeddings model selection
- System prompt customization

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure you have Ollama installed and running: `ollama serve`
4. Download the required models: `ollama pull nemotron-3-nano:4b` and `ollama pull ordis/jina-embeddings-v2-base-code:latest`

## Usage

Run the application:

```bash
python main.py
```

### Commands

- `/help` - Show help menu
- `/model` - Show or change the Ollama chat model
- `/embedding` - Show or change the Ollama embeddings model
- `/models` - List installed Ollama models
- `/select-model` - Select and persist an installed Ollama chat model
- `/select-embedding` - Select and persist an installed Ollama embeddings model
- `/system` - Show or set the system prompt
- `/system-reset` - Reset the system prompt
- `/stream` - Enable streaming
- `/no-stream` - Disable streaming
- `/history` - Show current session history
- `/reset` - Clear current session history
- `/about` - Show app info
- `/exit` - Exit app

### Keybindings

- `Ctrl-O` - Toggle thinking display
- `Ctrl-C` - Cancel current input (use `/exit` to quit)

## Configuration

Configuration is saved to `~/.simpleagent-cli/config.json`. You can manually edit this file to set default models and other settings.

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or feedback, please open an issue in this repository.