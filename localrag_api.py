import torch
import ollama
import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Configurações de cores (não usadas no servidor, mas mantidas por referência)
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

app = FastAPI()

# Supondo que index.html esteja na mesma pasta que o script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Permitir CORS para o frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajuste para o domínio do seu frontend em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa cliente OpenAI/Ollama
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3'
)

# Carrega vault.txt e gera embeddings uma vez no startup
vault_content = []
vault_embeddings = []

if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = [line.strip() for line in vault_file if line.strip()]

print("Gerando embeddings para o vault...")
for content in vault_content:
    resp = ollama.embeddings(model='mxbai-embed-large', prompt=content)
    vault_embeddings.append(resp["embedding"])
vault_embeddings_tensor = torch.tensor(vault_embeddings) if vault_embeddings else torch.tensor([])

def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=5):
    if vault_embeddings.nelement() == 0:
        return []
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.

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
        n=1,
        temperature=0.1,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return rewritten_query

class ChatRequest(BaseModel):
    user_input: str
    conversation_history: list

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    ollama_model = "llama3"  # fixado, pode parametrizar
    conversation_history = request.conversation_history or []
    user_input = request.user_input

    # Adiciona a mensagem do usuário ao histórico temporariamente
    conversation_history.append({"role": "user", "content": user_input})

    if len(conversation_history) > 1:
        query_json = json.dumps({"Query": user_input})
        rewritten_query = rewrite_query(query_json, conversation_history, ollama_model)
    else:
        rewritten_query = user_input

    relevant_context = get_relevant_context(rewritten_query, vault_embeddings_tensor, vault_content)

    user_input_with_context = user_input
    if relevant_context:
        context_str = "\n".join(relevant_context)
        user_input_with_context += f"\n\nRelevant Context:\n{context_str}"

    # Atualiza a última mensagem com o contexto
    conversation_history[-1]["content"] = user_input_with_context

    messages = [{"role": "system", "content": "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant information to the user query from outside the given context."}]
    messages.extend(conversation_history)

    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )
    assistant_reply = response.choices[0].message.content

    # Adiciona a resposta do assistente ao histórico
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return {
        "response": assistant_reply,
        "context": relevant_context,
        "conversation_history": conversation_history,
    }
