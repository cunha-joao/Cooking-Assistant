import torch
import ollama
import os
import json
from openai import OpenAI
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import argparse
from werkzeug.utils import secure_filename

# ========== COLOR CODES ==========
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# ========== FLASK APP ==========
app = Flask(__name__, static_folder='static', static_url_path='/')
CORS(app)

# ========== GLOBAL VARIABLES ==========
args = None
client = None
vault_embeddings_tensor = None
vault_content = []
system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in any extra relevant information to the user query from outside the given context."

# ========== UTILS ==========
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=5):
    if vault_embeddings is None or vault_embeddings.nelement() == 0:
        return []

    input_response = ollama.embeddings(model='nomic-embed-text', prompt=rewritten_input)
    input_embedding = input_response.get("embedding", [])

    if not input_embedding:
        print(YELLOW + "⚠️ Input embedding is empty. Skipping context retrieval." + RESET_COLOR)
        return []

    input_tensor = torch.tensor(input_embedding).unsqueeze(0)

    if input_tensor.size(1) != vault_embeddings.size(1):
        print(YELLOW + f"⚠️ Dimension mismatch: input {input_tensor.size()} vs vault {vault_embeddings.size()}" + RESET_COLOR)
        return []

    cos_scores = torch.cosine_similarity(input_tensor, vault_embeddings)
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context


def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
The rewritten query should:
- Preserve the core intent and meaning of the original query
- Expand and clarify the query to make it more specific and informative for retrieving relevant context
- Avoid introducing new topics or queries that deviate from the original query
- DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query

Return ONLY the rewritten query text, without any additional formatting or explanations.

Conversation History:
{context}

Original query: [{user_input}]

Rewritten query: 
"""
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        temperature=0.1,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})

def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})

    if len(conversation_history) > 1:
        query_json = {"Query": user_input}
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        print(PINK + "Original Query: " + user_input + RESET_COLOR)
        print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
    else:
        rewritten_query = user_input

    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)

    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context += "\n\nRelevant Context:\n" + context_str

    conversation_history[-1]["content"] = user_input_with_context
    messages = [{"role": "system", "content": system_message}, *conversation_history]

    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )

    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    return response.choices[0].message.content

# ========== FLASK ENDPOINT ==========
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.json
    user_input = data.get("user_input")
    conversation_history = data.get("conversation_history", [])

    response_text = ollama_chat(
        user_input=user_input,
        system_message=system_message,
        vault_embeddings=vault_embeddings_tensor,
        vault_content=vault_content,
        ollama_model=args.model,
        conversation_history=conversation_history
    )

    relevant_context = get_relevant_context(user_input, vault_embeddings_tensor, vault_content)

    return jsonify({
        "response": response_text,
        "context": relevant_context,
        "conversation_history": conversation_history
    })

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# ========== MAIN EXECUTION ==========
def load_embeddings():
    global vault_embeddings_tensor, vault_content
    vault_dir = "vaults"
    vault_content = []
    vault_embeddings = []

    print(NEON_GREEN + "Loading all vault files from 'vaults/'..." + RESET_COLOR)

    if not os.path.exists(vault_dir):
        print(YELLOW + "Directory 'vaults/' not found. Skipping loading." + RESET_COLOR)
        return

    for filename in os.listdir(vault_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(vault_dir, filename)
            print(f"{CYAN}Reading file: {file_path}{RESET_COLOR}")
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                vault_content.extend(lines)

    if not vault_content:
        print(YELLOW + "No content found in vaults." + RESET_COLOR)
        return

    print(NEON_GREEN + "Generating embeddings for vault content..." + RESET_COLOR)
    for content in vault_content:
        content = content.strip()
        if content:  # avoid blank lines
            response = ollama.embeddings(model='nomic-embed-text', prompt=content)
            vault_embeddings.append(response["embedding"])

    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    print(NEON_GREEN + f"Loaded {len(vault_content)} lines and generated embeddings." + RESET_COLOR)




@app.route('/change_vault', methods=['POST'])
def change_vault():
    global vault_embeddings_tensor, vault_content

    data = request.json
    filename = data.get('filename')

    vault_path = os.path.join("vaults", filename)
    print(f"Tentando abrir arquivo: {vault_path}")

    if not filename or not os.path.exists(vault_path):
        return jsonify({"success": False, "error": "Arquivo não encontrado"}), 400

    with open(vault_path, 'r', encoding='utf-8') as f:
        vault_content = f.readlines()

    vault_embeddings = []
    for content in vault_content:
        response = ollama.embeddings(model='nomic-embed-text', prompt=content)
        vault_embeddings.append(response["embedding"])

    vault_embeddings_tensor = torch.tensor(vault_embeddings)

    print(f"Vault trocado para {filename} e embeddings atualizados.")

    return jsonify({"success": True})


def start_cli():
    conversation_history = []
    while True:
        user_input = input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
        if user_input.lower() == 'quit':
            break
        response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
        print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)

def chat_from_web(user_input):
    global vault_embeddings_tensor, vault_content, conversation_history, system_message, args
    response = ollama_chat(
        user_input,
        system_message,
        vault_embeddings_tensor,
        vault_content,
        args.model,
        conversation_history
    )
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local RAG Chatbot (Flask)")
    parser.add_argument("--model", default="llama3", help="Model to use with Ollama")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli", help="Run in CLI or API mode")
    parser.add_argument("--port", type=int, default=8000, help="Port to run Flask app")
    args = parser.parse_args()

    print(NEON_GREEN + "Initializing OpenAI client..." + RESET_COLOR)
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="qwen3")

    load_embeddings()

    if args.mode == "cli":
        start_cli()
    else:
        print(NEON_GREEN + f"Running Flask server on port {args.port}..." + RESET_COLOR)
        app.run(host="0.0.0.0", port=args.port)
