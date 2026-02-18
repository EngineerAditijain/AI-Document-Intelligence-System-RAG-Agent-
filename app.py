# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import requests
# import re

# app = FastAPI()

# # ✅ CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5173",
#         "http://127.0.0.1:5173"
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# print("Loading embedding model...")
# embed_model = SentenceTransformer("BAAI/bge-small-en")

# print("Reading document...")
# with open("novel.txt", "r", encoding="utf-8") as f:
#     text = f.read()

# # 🔥 BETTER CHUNKING (split by principle)
# pattern = r"\d+\.\s.*?(?=\n\d+\.|\Z)"
# raw_chunks = re.findall(pattern, text, re.S)

# chunks = []
# for chunk in raw_chunks:
#     cleaned = chunk.strip()
#     if cleaned:
#         chunks.append(cleaned)

# print(f"Created {len(chunks)} chunks")

# print("Creating embeddings...")
# embeddings = embed_model.encode(chunks, normalize_embeddings=True)

# dimension = embeddings.shape[1]
# index = faiss.IndexFlatIP(dimension)
# index.add(np.array(embeddings))

# print("Vector store ready.")

# class Query(BaseModel):
#     question: str

# @app.post("/ask")
# def ask(query: Query):

#     query_embedding = embed_model.encode(
#         [query.question],
#         normalize_embeddings=True
#     )

#     # 🔥 Retrieve TOP 3 instead of 1
#     D, I = index.search(query_embedding, k=3)

#     print("---- Retrieved Chunks ----")
#     retrieved_chunks = []
#     for score, idx in zip(D[0], I[0]):
#         print("Score:", score)
#         print(chunks[idx])
#         print("------")
        
#         # optional similarity filter
#         if score > 0.4:
#             retrieved_chunks.append(chunks[idx])

#     if not retrieved_chunks:
#         return {"answer": "Answer not found in the document."}

#     context = "\n\n".join(retrieved_chunks)

#     prompt = f"""
# You are a strict document-based question answering system.

# RULES:
# - Only answer using the DOCUMENT.
# - Do not summarize unless asked.
# - Do not explain extra.
# - If answer is not clearly in DOCUMENT, say:
#   "Answer not found in the document."

# DOCUMENT:
# ----------------
# {context}
# ----------------

# QUESTION:
# {query.question}

# FINAL ANSWER:
# """

#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": "tinyllama",
#             "prompt": prompt,
#             "stream": False,
#             "options": {
#                 "temperature": 0,
#                 "num_predict": 200
#             }
#         }
#     )

#     return {"answer": response.json()["response"].strip()}


# @app.get("/health")
# def health():
#     return {"status": "ok"}
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import re
import json
import time

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading embedding model...")
embed_model = SentenceTransformer("BAAI/bge-small-en")

print("Reading document...")
with open("novel.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 🔥 BETTER CHUNKING (split by principle)
pattern = r"\d+\.\s.*?(?=\n\d+\.|\Z)"
raw_chunks = re.findall(pattern, text, re.S)

chunks = []
for chunk in raw_chunks:
    cleaned = chunk.strip()
    if cleaned:
        # 🔥 Enrich chunk for better retrieval
        title = cleaned.split("\n")[0]
        enriched = f"This is {title}.\n\n{cleaned}"
        chunks.append(enriched)

print(f"Created {len(chunks)} chunks")

print("Creating embeddings...")
embeddings = embed_model.encode(chunks, normalize_embeddings=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(np.array(embeddings))

print("Vector store ready.")


class Query(BaseModel):
    question: str


# =========================
# 🔎 RAG ENDPOINT
# =========================
@app.post("/ask")
def ask(query: Query):

    query_embedding = embed_model.encode(
        [query.question],
        normalize_embeddings=True
    )

    # Retrieve top 5 for better accuracy
    D, I = index.search(query_embedding, k=5)

    print("---- Retrieved Chunks ----")
    retrieved_chunks = []

    for score, idx in zip(D[0], I[0]):
        print("Score:", score)
        print(chunks[idx])
        print("------")

        if score > 0.4:
            retrieved_chunks.append(chunks[idx])

    if not retrieved_chunks:
        return {"answer": "Answer not found in the document."}

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a strict document-based question answering system.

RULES:
- Only answer using the DOCUMENT.
- Do not summarize unless asked.
- Do not explain extra.
- If answer is not clearly in DOCUMENT, say:
  "Answer not found in the document."

DOCUMENT:
----------------
{context}
----------------

QUESTION:
{query.question}

FINAL ANSWER:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 200
            }
        }
    )

    return {"answer": response.json()["response"].strip()}


# =========================
# 🤖 AGENT TOOLS
# =========================

def calculator(expression: str):
    try:
        return str(eval(expression))
    except:
        return "Invalid math expression"

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def search_document(query_text: str):
    query_embedding = embed_model.encode(
        [query_text],
        normalize_embeddings=True
    )

    D, I = index.search(query_embedding, k=3)

    results = []
    for score, idx in zip(D[0], I[0]):
        if score > 0.4:
            results.append(chunks[idx])

    return "\n\n".join(results) if results else "No relevant document content found."


TOOLS = {
    "calculator": calculator,
    "get_time": get_time,
    "search_document": search_document
}


# =========================
# 🧠 REAL THINKING AGENT
# =========================
@app.post("/agent")
def run_agent(query: Query):

    question = query.question.strip()

    # 🔒 HARD ENFORCEMENT: Extract math expression only
    math_match = re.search(r"(\d+\s*[\+\-\*\/]\s*\d+)", question)

    if math_match:
        expression = math_match.group(1)
        result = calculator(expression)
        return {"answer": result}

    messages = [
        {
            "role": "system",
            "content": """
You are an autonomous AI agent.

IMPORTANT:
- You MUST respond ONLY in valid JSON.
- Do NOT write explanations.
- Do NOT write text outside JSON.
- If math is needed, use calculator tool.
- If document info is needed, use search_document tool.

TOOLS:
1. calculator(expression)
2. get_time()
3. search_document(query)

Tool usage format:
{
  "action": "tool_name",
  "input": "tool_input"
}

Final answer format:
{
  "final_answer": "your answer"
}
"""
        }
    ]

    messages.append({"role": "user", "content": question})

    for step in range(5):

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "tinyllama",  # ✅ kept tinyllama
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0
                }
            }
        )

        output = response.json()["message"]["content"].strip()
        print("LLM OUTPUT:", output)

        # 🔥 Attempt to extract JSON even if model adds noise
        try:
            json_start = output.find("{")
            json_end = output.rfind("}") + 1
            cleaned_json = output[json_start:json_end]

            parsed = json.loads(cleaned_json)

        except:
            return {"answer": "Agent failed to produce valid JSON."}

        # ✅ Final answer
        if "final_answer" in parsed:
            return {"answer": parsed["final_answer"]}

        # ✅ Tool call
        if "action" in parsed:
            tool_name = parsed["action"]
            tool_input = parsed.get("input", "")

            if tool_name in TOOLS:
                result = TOOLS[tool_name](tool_input)

                messages.append({
                    "role": "assistant",
                    "content": cleaned_json
                })

                messages.append({
                    "role": "tool",
                    "content": result
                })
            else:
                return {"answer": "Unknown tool requested."}

    return {"answer": "Agent stopped after max reasoning steps."}

@app.get("/health")
def health():
    return {"status": "ok"}
