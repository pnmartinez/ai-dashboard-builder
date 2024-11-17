#!/bin/sh

# Start the Ollama server in the background
ollama serve  &

# Wait for the server to start
sleep 10

# Pull the required model
ollama pull llama3.1

# Keep the container running
tail -f /dev/null
