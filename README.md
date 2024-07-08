Ollama-friendly OpenAI Embeddings Proxy. This script bridges the gap between OpenAI's embedding API and Ollama, making it compatible with the current version of Graphrag.
To use the script, run it and then update the embeddings section in your Graphrag settings.yaml file like this:
YAML
model: nomic-embed-text
api_base: http://localhost:5000/v1/
