import os
import uuid
import re
import fitz  # PyMuPDF
from sympy import re
import tiktoken
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile , File , HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llm_client import get_client
from google import genai as google_genai
from google.genai import types as genai_types
import time

load_dotenv()
# Use environment variable to switch providers (default: gemini)
llm_client = get_client(os.getenv("LLM_PROVIDER", "gemini"))
gemini_client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
# Remove file if it exists (in case uploads is a file instead of directory)
if os.path.exists(UPLOAD_DIR) and os.path.isfile(UPLOAD_DIR):
    os.remove(UPLOAD_DIR)
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    session_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "saved_as": save_path
    }

@app.get("/health")
def health():
    return {"status": "ok"}

def extract_text_from_pdf(pdf_path : str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


@app.get("/retrieve")
async def retrieve_chunks(session_id: str, q: str, top_k: int = 3):
    try:
        collection = chroma_client.get_collection(name=session_id)
    except Exception:
        raise HTTPException(status_code=404, detail="session not found. run /embed first to create collection.")
    
    query_embedding = embedding_model.encode([q]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count())
    )

    chunks = results['documents'][0]
    distances = results['distances'][0]

    return {
        "query": q,
        "results": [
            {"chunk": chunk, "similarity_score": round(1 - dist, 4)}
            for chunk, dist in zip(chunks, distances)
        ]
    }
          


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap #sliding window with overlap

    return chunks

@app.post("/process")
async def process_pdf(session_id: str):
    pdf_path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found for the given session_id.")
    
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    return {
        "session_id": session_id,
        "total_chunks": len(chunks),
        "total_chars": len(text),
        "preview": chunks[:2]  # Return first 2 chunks as a preview
    }


def embed_and_store(session_id: str, chunks: list[str]):
    collection = chroma_client.get_or_create_collection(name=session_id)

    embeddings = embedding_model.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids = [f"{session_id}_{i}" for i in range(len(chunks))]
    )

    return collection

@app.post("/embed")
async def embed_pdf(session_id: str):
    pdf_path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found.")

    text = extract_text_from_pdf(pdf_path)
    text = scrub_pii(text)          # scrub before chunking
    chunks = chunk_text(text)
    collection = embed_and_store(session_id, chunks)

    return {
        "session_id": session_id,
        "chunks_embedded": collection.count(),
        "status": "stored in chromadb"
    }


def build_prompt(question_topic: str, context_chunks: list[str]) -> str:
    context = "\n--\n".join(context_chunks)
    return f"""You are a technical interviewer. Use the candidate context below to generate 3 spectific interview questions about "{question_topic}". Only use the candidate context to generate questions. If the context is not relevant to the topic, say "not enough information to generate questions".

    CANDIDATE CONTEXT:
    {context}

    Rules:
    - Questions must reference specific details from the context
    -No generic questions like "tell me about yourself"
    -Return excactly 3 questions, numbered 1-3
    """

@app.get("/generate-questions")
async def generate_questions(session_id: str, topic: str):
    # sanitize input
    topic = sanitize_input(topic, max_length=200)

    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty after sanitization.")

    try:
        try:
            collection = chroma_client.get_collection(name=session_id)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Session not found. Run /embed first.")

        query_embedding = embedding_model.encode([topic]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(3, collection.count())
        )
        chunks = results['documents'][0]

        prompt_with_rag = build_prompt(topic, chunks)
        prompt_without_rag = build_prompt(topic, [])

        try:
            response_with = llm_client.chat(prompt_with_rag)
            time.sleep(13)
            response_without = llm_client.chat(prompt_without_rag)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

        return {
            "topic": topic,
            "with_rag": response_with["text"],
            "without_rag": response_without["text"],
            "chunks_used": chunks
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    


def sanitize_input(text: str, max_length: int = 500) -> str:
    # strip null bytes
    text = text.replace("\x00", "")
    # remove control characters
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # truncate to max length
    text = text[:max_length]
    return text.strip()

def scrub_pii(text: str) -> str:
    """Remove common PII patterns before storing or returning chunks."""
    # email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', text)
    # phone numbers (Indian + international formats)
    text = re.sub(r'(\+91[\-\s]?)?[6-9]\d{9}', '[PHONE]', text)
    # LinkedIn/GitHub URLs with usernames
    text = re.sub(r'(linkedin\.com/in/|github\.com/)[\w\-]+', r'\1[USERNAME]', text)
    return text