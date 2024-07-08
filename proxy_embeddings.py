import flask
from flask import request, jsonify
import requests
import logging
import sys
from werkzeug.exceptions import HTTPException
import tiktoken

app = flask.Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/embeddings"

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")  # This is the encoding used by text-embedding-ada-002

@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e
    logger.error(f"An unhandled exception occurred: {str(e)}", exc_info=True)
    return jsonify(error="An unexpected error occurred"), 500

def detokenize(token_ids):
    return tokenizer.decode(token_ids)

def process_single_input(input_data):
    if isinstance(input_data, list) and all(isinstance(item, int) for item in input_data):
        input_text = detokenize(input_data)
    elif isinstance(input_data, str):
        input_text = input_data
    else:
        raise ValueError(f"Invalid input type: {type(input_data)}")

    ollama_data = {
        "model": "nomic-embed-text",
        "prompt": input_text
    }
    logger.info(f"Sending request to Ollama: {ollama_data}")
    
    try:
        response = requests.post(OLLAMA_API_URL, json=ollama_data, timeout=30)
        logger.debug(f"Ollama API response status code: {response.status_code}")
        logger.debug(f"Ollama API response content: {response.text}")
        response.raise_for_status()
        ollama_response = response.json()
        return ollama_response['embedding']
    except requests.RequestException as e:
        logger.error(f"Error communicating with Ollama: {str(e)}")
        logger.error(f"Ollama response content: {e.response.text if e.response else 'No response content'}")
        raise

@app.route('/v1/embeddings', methods=['POST'])
def proxy_openai_to_ollama():
    try:
        data = request.json
        logger.info(f"Received request: {data}")

        if 'input' not in data:
            logger.error("Missing 'input' in request data")
            return jsonify(error="Missing 'input' in request"), 400

        input_data = data['input']
        
        if isinstance(input_data, (str, list)):
            embeddings = [process_single_input(input_data)]
        elif isinstance(input_data, list) and all(isinstance(item, (str, list)) for item in input_data):
            embeddings = [process_single_input(item) for item in input_data]
        else:
            logger.error(f"Invalid input type: {type(input_data)}")
            return jsonify(error="Invalid input type. Expected string, array of integers, or array of strings/arrays."), 400

        openai_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": emb,
                    "index": idx
                } for idx, emb in enumerate(embeddings)
            ],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": len(input_data) if isinstance(input_data[0], int) else sum(len(text) for text in input_data),
                "total_tokens": len(input_data) if isinstance(input_data[0], int) else sum(len(text) for text in input_data)
            }
        }
        logger.info("Successfully processed request and generated response")

        return jsonify(openai_response)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        return jsonify(error="An unexpected error occurred"), 500

if __name__ == '__main__':
    logger.info("Starting the OpenAI to Ollama Embedding API Proxy")
    app.run(port=5000)