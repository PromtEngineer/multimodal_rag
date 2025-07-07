#!/usr/bin/env python3
"""
Script to create and build indexes using the backend API
"""

import requests
import json
import time
import os
import sys
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"
DOCUMENTS_DIR = "rag_system/documents"

def check_backend_health():
    """Check if the backend server is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def create_index(name, description=""):
    """Create a new index"""
    print(f"📝 Creating index: {name}")
    
    data = {
        "name": name,
        "description": description
    }
    
    response = requests.post(f"{BACKEND_URL}/indexes", json=data)
    
    if response.status_code == 201:
        result = response.json()
        index_id = result["index_id"]
        print(f"✅ Index created successfully with ID: {index_id}")
        return index_id
    else:
        print(f"❌ Failed to create index: {response.text}")
        return None

def upload_documents(index_id, file_paths):
    """Upload documents to an index"""
    print(f"📤 Uploading {len(file_paths)} documents to index {index_id}")
    
    files = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            files.append(('files', (os.path.basename(file_path), open(file_path, 'rb'))))
        else:
            print(f"⚠️ File not found: {file_path}")
    
    if not files:
        print("❌ No valid files to upload")
        return False
    
    try:
        response = requests.post(f"{BACKEND_URL}/indexes/{index_id}/upload", files=files)
        
        # Close file handles
        for _, (_, file_handle) in files:
            file_handle.close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Uploaded {len(result['uploaded_files'])} files successfully")
            for file_info in result['uploaded_files']:
                print(f"   - {file_info['filename']}")
            return True
        else:
            print(f"❌ Failed to upload files: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error uploading files: {e}")
        return False

def build_index(index_id):
    """Build/process the index"""
    print(f"🔨 Building index {index_id}")
    
    response = requests.post(f"{BACKEND_URL}/indexes/{index_id}/build")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Index built successfully!")
        
        # Show build configuration
        if 'response' in result and 'indexing_config' in result['response']:
            config = result['response']['indexing_config']
            print("📊 Index Configuration:")
            print(f"   - Chunk Size: {config.get('chunk_size', 'N/A')}")
            print(f"   - Chunk Overlap: {config.get('chunk_overlap', 'N/A')}")
            print(f"   - Retrieval Mode: {config.get('retrieval_mode', 'N/A')}")
            print(f"   - Window Size: {config.get('window_size', 'N/A')}")
            print(f"   - Enable Enrich: {config.get('enable_enrich', 'N/A')}")
        
        return True
    else:
        print(f"❌ Failed to build index: {response.text}")
        return False

def get_index_status(index_id):
    """Get the status and metadata of an index"""
    response = requests.get(f"{BACKEND_URL}/indexes/{index_id}")
    
    if response.status_code == 200:
        result = response.json()
        metadata = result.get('metadata', {})
        
        print(f"📋 Index Status for {index_id}:")
        print(f"   - Status: {metadata.get('status', 'Unknown')}")
        print(f"   - Embedding Model: {metadata.get('embedding_model', 'N/A')}")
        print(f"   - Chunk Size: {metadata.get('chunk_size', 'N/A')}")
        print(f"   - Chunk Overlap: {metadata.get('chunk_overlap', 'N/A')}")
        print(f"   - Retrieval Mode: {metadata.get('retrieval_mode', 'N/A')}")
        print(f"   - Enrichment Enabled: {metadata.get('enable_enrich', 'N/A')}")
        
        return metadata
    else:
        print(f"❌ Failed to get index status: {response.text}")
        return None

def list_available_documents():
    """List available documents in the documents directory"""
    docs_path = Path(DOCUMENTS_DIR)
    if not docs_path.exists():
        print(f"❌ Documents directory not found: {DOCUMENTS_DIR}")
        return []
    
    documents = []
    for ext in ['*.pdf', '*.txt', '*.docx', '*.md']:
        documents.extend(docs_path.glob(ext))
    
    return [str(doc) for doc in documents]

def main():
    print("🚀 Index Creation Script")
    print("=" * 50)
    
    # Check if backend is running
    if not check_backend_health():
        print("❌ Backend server is not running!")
        print("Please start the backend server first:")
        print("   cd backend && python server.py")
        sys.exit(1)
    
    print("✅ Backend server is running")
    
    # Get available documents
    available_docs = list_available_documents()
    if not available_docs:
        print("❌ No documents found in the documents directory")
        sys.exit(1)
    
    print(f"📁 Found {len(available_docs)} documents:")
    for i, doc in enumerate(available_docs, 1):
        print(f"   {i}. {os.path.basename(doc)}")
    
    # Get user input
    index_name = input("\n📝 Enter index name: ").strip()
    if not index_name:
        print("❌ Index name is required")
        sys.exit(1)
    
    index_description = input("📝 Enter index description (optional): ").strip()
    
    # Select documents to upload
    print("\n📤 Select documents to upload:")
    print("   Enter document numbers separated by commas (e.g., 1,2,3)")
    print("   Or press Enter to upload all documents")
    
    doc_selection = input("Documents: ").strip()
    
    if doc_selection:
        try:
            selected_indices = [int(x.strip()) - 1 for x in doc_selection.split(',')]
            selected_docs = [available_docs[i] for i in selected_indices if 0 <= i < len(available_docs)]
        except (ValueError, IndexError):
            print("❌ Invalid document selection")
            sys.exit(1)
    else:
        selected_docs = available_docs
    
    print(f"\n🎯 Selected {len(selected_docs)} documents for upload")
    
    # Create the index
    index_id = create_index(index_name, index_description)
    if not index_id:
        sys.exit(1)
    
    # Upload documents
    if not upload_documents(index_id, selected_docs):
        sys.exit(1)
    
    # Build the index
    if not build_index(index_id):
        sys.exit(1)
    
    # Show final status
    print("\n" + "=" * 50)
    get_index_status(index_id)
    
    print(f"\n🎉 Index '{index_name}' created successfully!")
    print(f"   Index ID: {index_id}")
    print(f"   You can now use this index in your chat sessions.")

if __name__ == "__main__":
    main() 