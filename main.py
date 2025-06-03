from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# === Initialize FastAPI ===
app = FastAPI()

# === CORS setup (important for public API use) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load models ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# === Request body structure ===
class QARequest(BaseModel):
    url: str
    question: str

# === Cache for embeddings per URL ===
cache = {}

# === Scrape website and clean text ===
def scrape_text(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove unnecessary tags
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "iframe"]):
        tag.decompose()

    # Try to get just the article content
    article_tags = soup.find_all(['p', 'h1', 'h2'])
    article_text = " ".join(tag.get_text(separator=" ", strip=True) for tag in article_tags)

    # Fallback if no good tags found
    if not article_text or len(article_text) < 100:
        article_text = soup.get_text(separator=" ", strip=True)

    # Remove repeated or meaningless words like "MSN MSN MSN"
    words = article_text.split()
    cleaned_words = []
    for i in range(len(words)):
        if i == 0 or words[i] != words[i - 1]:
            cleaned_words.append(words[i])
    cleaned_text = " ".join(cleaned_words)

    # Remove excessive whitespace and short noise
    final_text = " ".join(w for w in cleaned_text.split() if len(w) > 2)
    return final_text.strip()


# === Break text into chunks and embed them ===
def embed_chunks(text, chunk_size=500):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks

# === Retrieve most relevant chunks ===
def search_chunks(index, chunks, question, top_k=5):
    q_embedding = embedding_model.encode([question])
    distances, indices = index.search(np.array(q_embedding), top_k)

    seen = set()
    unique_chunks = []
    for i in indices[0]:
        chunk = chunks[i].strip()
        if chunk not in seen:
            unique_chunks.append(chunk)
            seen.add(chunk)
    return unique_chunks

# === Generate detailed answer from context and question ===
def generate_answer(context, question):
    prompt = f"""
You are an intelligent assistant. Based on the news article context below, answer the question in detail and avoid repeating words.

Context:
{context}

Question: {question}
Answer:"""
    result = qa_pipeline(prompt, max_new_tokens=300)
    return result[0]["generated_text"].strip()

# === Main API Endpoint ===
@app.post("/ask")
def ask(data: QARequest):
    url = data.url
    question = data.question

    if url not in cache:
        text = scrape_text(url)
        index, chunks = embed_chunks(text)
        cache[url] = (index, chunks)
    else:
        index, chunks = cache[url]

    relevant_chunks = search_chunks(index, chunks, question)
    combined_context = " ".join(relevant_chunks)
    answer = generate_answer(combined_context, question)

    return {"answer": answer}
