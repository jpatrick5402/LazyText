# Use the official Ollama image as the base
FROM ollama/ollama:latest

# Set environment variables for Ollama
ENV OLLAMA_HOST="0.0.0.0"

# Pull the Gemma 4 model during the build
RUN ollama serve >/tmp/ollama.log 2>&1 & \
    pid=$!; \
    for i in $(seq 1 30); do \
      curl -sf http://127.0.0.1:11434/api/tags && break; \
      sleep 1; \
    done; \
    ollama pull gemma4; \
    kill $pid; \
    wait $pid || true

# Expose the default port for Ollama
EXPOSE 11434
