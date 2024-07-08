import flask
from flask import request, jsonify
import requests
import logging
import sys
from werkzeug.exceptions import HTTPException
import tiktoken
import json

app = flask.Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Ensure Flask's logger is also set to DEBUG
app.logger.setLevel(logging.DEBUG)

OLLAMA_API_URL = "http://localhost:11434/api/embeddings"

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")  # This is the encoding used by text-embedding-ada-002

@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e
    logger.error(f"An unhandled exception occurred: {str(e)}", exc_info=True)
    return jsonify(error="An unexpected error occurred"), 500

def process_single_input(input_item):
    # Convert input_item to string if it's not already
    input_item = str(input_item)
    
    logger.debug(f"Processing single input: {input_item[:100]}...")  # Log first 100 chars
    
    # Send request to Ollama
    response = requests.post(OLLAMA_API_URL, json={
        "model": "nomic-embed-text",
        "prompt": input_item
    })
    
    logger.debug(f"Ollama API response status code: {response.status_code}")
    
    if response.status_code == 200:
        embedding = response.json().get('embedding')
        if embedding:
            logger.debug(f"Received embedding of length: {len(embedding)}")
            logger.debug(f"First few values of embedding: {embedding[:5]}...")
            return embedding
        else:
            logger.error("No embedding found in Ollama API response")
            return None
    else:
        logger.error(f"Ollama API request failed with status code {response.status_code}")
        return None

def process_input(input_data):
    logger.debug(f"process_input received: type={type(input_data)}")
    logger.debug(f"Input data (truncated): {str(input_data)[:500]}...")
    
    if isinstance(input_data, list):
        embeddings = []
        for item in input_data:
            embedding = process_single_input(item)
            if embedding:
                embeddings.append(embedding)
        return embeddings
    else:
        return [process_single_input(input_data)]

@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    data = request.json
    logger.info(f"Received request: {json.dumps(data, indent=2)}")
    
    input_data = data.get('input', [])
    logger.debug(f"Input data: type={type(input_data)}")
    logger.debug(f"Input data (truncated): {str(input_data)[:500]}...")
    
    if isinstance(input_data, list):
        logger.debug(f"Number of input items: {len(input_data)}")
        logger.debug(f"First few input items: {input_data[:5]}")
    else:
        input_data = [input_data]
    
    embeddings = process_input(input_data)
    
    response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": emb,
                "index": i
            } for i, emb in enumerate(embeddings)
        ],
        "model": "text-embedding-ada-002",
        "usage": {
            "prompt_tokens": sum(len(tokenizer.encode(str(item))) for item in input_data),
            "total_tokens": sum(len(tokenizer.encode(str(item))) for item in input_data)
        }
    }
    
    logger.info(f"Successfully processed request and generated response with {len(embeddings)} embeddings")
    logger.debug(f"Response structure: {json.dumps(response, indent=2, default=str)}")
    logger.debug(f"First embedding in response (first few values): {embeddings[0][:5]}...")
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
