# MCP Chat Examples with Conversation Memory

This project demonstrates how to create chat applications using MCP (Model Context Protocol) agents with built-in conversation memory. The examples integrate with your configured MCP servers for enhanced capabilities.

## Features

- 🤖 **MCP Agent Integration**: Connects to your configured MCP servers (Playwright, Airbnb, DuckDuckGo Search)
- 💭 **Conversation Memory**: Built-in memory using LangChain's ConversationBufferMemory
- 🔄 **Async Support**: Full async/await support for better performance
- 📝 **Chat History**: View and manage conversation history
- 🚀 **Groq LLM Integration**: Uses Groq's fast LLM for responses

## MCP Servers Configured

Your `browser_mcp.json` includes:
- **playwright**: Web automation and browsing capabilities
- **airbnb**: Travel and accommodation information
- **duckduckgo-search**: Web search functionality

## Setup

### 1. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using uv (recommended)
uv pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
# Set your Groq API key
export GROQ_API_KEY="your-groq-api-key-here"

# On Windows PowerShell
$env:GROQ_API_KEY="your-groq-api-key-here"
```

### 3. Get a Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up or log in
3. Create an API key
4. Copy the key to your environment variables

## Usage

### Simple Chat Example

Run the streamlined chat example:

```bash
python simple_chat_example.py
```

### Full Featured Chat Example

Run the comprehensive chat example:

```bash
python app.py
```

### Interactive Commands

- **Type your message**: Just type and press Enter
- **`history`**: View chat history
- **`quit`**: Exit the chat

## Example Conversation

```
🤖 Simple MCP Chat with Memory
========================================
Commands: 'history', 'quit'
========================================

👤 You: Hello! What can you help me with?

🤖 Thinking...
🤖 Assistant: Hello! I'm your AI assistant with access to several MCP servers. I can help you with:

- Web browsing and automation (via Playwright)
- Travel and accommodation information (via Airbnb)
- Web search capabilities (via DuckDuckGo)

What would you like to explore today?

👤 You: Can you help me search for something?

🤖 Thinking...
🤖 Assistant: Absolutely! I have access to DuckDuckGo search capabilities through my MCP server. I can help you search the web for any information you need. What would you like me to search for?
```

## Project Structure

```
firstproject/
├── app.py                    # Full-featured MCP chat example
├── simple_chat_example.py   # Streamlined chat example
├── browser_mcp.json         # MCP server configuration
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Customization

### Adding New MCP Servers

Edit `browser_mcp.json` to add new servers:

```json
{
  "mcpServers": {
    "your-server": {
      "command": "npx",
      "args": ["-y", "your-mcp-package"]
    }
  }
}
```

### Modifying Memory Behavior

The examples use LangChain's `ConversationBufferMemory`. You can modify the memory implementation in the code to use different memory types like:
- `ConversationSummaryMemory`
- `ConversationTokenBufferMemory`
- `ConversationBufferWindowMemory`

## Troubleshooting

### Common Issues

1. **GROQ_API_KEY not set**: Make sure you've set the environment variable
2. **MCP servers not loading**: Check that `browser_mcp.json` exists and is valid JSON
3. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

### Debug Mode

Add debug prints to see what's happening:

```python
# In the chat methods, add:
print(f"Debug: Processing input: {user_input}")
print(f"Debug: Context: {context}")
```

## Next Steps

- Integrate actual MCP server calls in responses
- Add streaming responses
- Implement conversation export/import
- Add user authentication
- Create a web interface

## License

This project is open source. Feel free to modify and distribute as needed.
