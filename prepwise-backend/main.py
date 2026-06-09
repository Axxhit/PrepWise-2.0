import os
import uuid
import fitz  # PyMuPDF
import tiktoken
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile , File , HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-3.5-flash")

app = FastAPI()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()

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
        raise HTTPException(status_code=404, detail="PDF not found for the given session_id.")
    
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    collection = embed_and_store(session_id, chunks)

    return {
        "session_id": session_id,
        "total_chunks": len(chunks),
        "collection_name": collection.name
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

@app.get("/generate_questions")
async def generate_questions(session_id: str, topic: str):
    #step 1 - retrieve relevant chunks
    try:
        collection = chroma_client.get_collection(name=session_id)
    except Exception:
        raise HTTPException(status_code=404, detail="session not found. run /embed first to create collection.")
    
    query_embedding = embedding_model.encode([topic]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(3, collection.count())  # get top 3 most relevant chunks
    )
    chunks = results['documents'][0]

    #step 2 - build prompt and generate questions using Gemini with context
    prompt_with_rag = build_prompt(topic, chunks)
    prompt_without_rag = build_prompt(topic, [])  # empty context for ablation

    #step3 - call gemini with and without context for comparison
    response_with = gemini.generate_content(prompt_with_rag)
    response_without = gemini.generate_content(prompt_without_rag)

    return {
        "topic": topic,
        "with_rag": response_with.text,
        "without_rag": response_without.text,
        "chunks_used": chunks
    }
