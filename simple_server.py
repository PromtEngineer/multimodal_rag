from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Dict, Any
import uuid
import os
import shutil

# Import existing function logic
from comprehensive_indexing import run_indexing_pipeline, create_indexing_config, create_ollama_config
from comprehensive_retrieval import test_agent_retrieval, create_retrieval_config

app = FastAPI()

# Upload directory
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-file/")
async def upload_file(file: UploadFile):
    upload_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

@app.post("/index/")
async def index_file(
    index_name: str = Form(...),
    chunk_size: int = Form(1024),
    embedding_model: str = Form("Qwen/Qwen3-Embedding-0.6B")
):
    # Sample configuration
    args_index = {
        'folder': UPLOAD_DIR,
        'index_name': index_name,
        'chunk_size': chunk_size,
        'embedding_model': embedding_model
    }

    config, index_id = create_indexing_config(args_index)
    ollama_config = create_ollama_config(args_index)
    result = run_indexing_pipeline(config, ollama_config, [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)], index_id)
    
    if result.get('success', False):
        return {"message": "Indexing successful", "index_id": index_id}
    else:
        return JSONResponse(status_code=500, content={"message": "Indexing failed", "details": result})

@app.post("/chat/")
async def chat(
    session_id: str = Form(...),
    query: str = Form(...),
    index_name: str = Form(...)
):
    # Use the test_agent_retrieval as a placeholder
    args_retrieval = {
        'lancedb_path': './lancedb',
        'enable_context_expansion': False,
        'enable_query_decomposition': True,
        'enable_reranker': True,
        'embedding_model': "Qwen/Qwen3-Embedding-0.6B"
    }

    config = create_retrieval_config(args_retrieval, index_name)
    ollama_config = create_ollama_config(args_retrieval)
    result = test_agent_retrieval(config, ollama_config, query, index_name, index_name)

    if result.get('success', False):
        return {"answer": result['answer'], "sources": result.get('source_documents', [])}
    else:
        return JSONResponse(status_code=500, content={"message": "Chat retrieval failed", "details": result})

