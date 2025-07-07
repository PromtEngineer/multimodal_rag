#!/bin/bash

# Simple Index Creation Script
# Usage: ./simple_create_index.sh "Index Name" "path/to/document.pdf"

BACKEND_URL="http://localhost:8000"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 \"Index Name\" \"path/to/document.pdf\""
    echo "Example: $0 \"My Index\" \"rag_system/documents/invoice_1039.pdf\""
    exit 1
fi

INDEX_NAME="$1"
DOCUMENT_PATH="$2"

echo "🚀 Creating Index: $INDEX_NAME"

# Step 1: Create index
echo "📝 Creating index..."
RESPONSE=$(curl -s -X POST "$BACKEND_URL/indexes" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"$INDEX_NAME\", \"description\": \"Created via script\"}")

INDEX_ID=$(echo "$RESPONSE" | jq -r '.index_id')

if [[ "$INDEX_ID" == "null" || "$INDEX_ID" == "" ]]; then
    echo "❌ Failed to create index: $RESPONSE"
    exit 1
fi

echo "✅ Index created with ID: $INDEX_ID"

# Step 2: Upload document
echo "📤 Uploading document..."
UPLOAD_RESPONSE=$(curl -s -X POST "$BACKEND_URL/indexes/$INDEX_ID/upload" \
    -F "files=@$DOCUMENT_PATH")

echo "Upload response: $UPLOAD_RESPONSE"

# Step 3: Build index
echo "🔨 Building index..."
BUILD_RESPONSE=$(curl -s -X POST "$BACKEND_URL/indexes/$INDEX_ID/build")

echo "Build response: $BUILD_RESPONSE"

# Step 4: Check status
echo "📋 Final status:"
curl -s "$BACKEND_URL/indexes/$INDEX_ID" | jq '.metadata | {status, chunk_size, chunk_overlap, retrieval_mode, embedding_model}'

echo "🎉 Index creation completed!"
echo "Index ID: $INDEX_ID" 