#!/usr/bin/env python3
"""
Comprehensive Test Suite for Consolidated RAG System
Tests all major functionality including session management, file uploads, indexing, and RAG queries.
"""

import sys
import os
import json
import time
import requests
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystemTester:
    """Comprehensive test suite for the consolidated RAG system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_session_id = None
        self.test_index_id = None
        self.test_files = []
        
    def wait_for_server(self, timeout: int = 60) -> bool:
        """Wait for server to be ready"""
        logger.info(f"Waiting for server at {self.base_url}...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Server is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        
        logger.error("❌ Server not ready within timeout")
        return False
    
    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        logger.info("🔍 Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Health check passed: {data.get('status')}")
                return True
            else:
                logger.error(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False
    
    def test_models_endpoint(self) -> bool:
        """Test models listing endpoint"""
        logger.info("🔍 Testing models endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/models")
            if response.status_code == 200:
                data = response.json()
                gen_models = data.get('generation_models', [])
                emb_models = data.get('embedding_models', [])
                logger.info(f"✅ Models endpoint: {len(gen_models)} generation, {len(emb_models)} embedding")
                return True
            else:
                logger.error(f"❌ Models endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Models endpoint error: {e}")
            return False
    
    def test_session_management(self) -> bool:
        """Test session creation and management"""
        logger.info("🔍 Testing session management...")
        try:
            # Create session
            create_data = {
                "title": "Test Session",
                "model": "qwen3:8b"
            }
            response = self.session.post(f"{self.base_url}/sessions", json=create_data)
            if response.status_code == 201:
                data = response.json()
                self.test_session_id = data['session_id']
                logger.info(f"✅ Session created: {self.test_session_id}")
            else:
                logger.error(f"❌ Session creation failed: {response.status_code}")
                return False
            
            # Get session
            response = self.session.get(f"{self.base_url}/sessions/{self.test_session_id}")
            if response.status_code == 200:
                logger.info("✅ Session retrieval successful")
            else:
                logger.error(f"❌ Session retrieval failed: {response.status_code}")
                return False
            
            # List sessions
            response = self.session.get(f"{self.base_url}/sessions")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Sessions listed: {data.get('total', 0)} sessions")
                return True
            else:
                logger.error(f"❌ Sessions listing failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Session management error: {e}")
            return False
    
    def test_file_upload(self) -> bool:
        """Test file upload functionality"""
        logger.info("🔍 Testing file upload...")
        try:
            # Create test file
            test_content = """
            This is a test document for the RAG system.
            It contains information about artificial intelligence and machine learning.
            The document discusses various topics including:
            - Natural language processing
            - Document retrieval systems
            - Question answering systems
            - Semantic search capabilities
            """
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                test_file_path = f.name
            
            self.test_files.append(test_file_path)
            
            # Upload file
            with open(test_file_path, 'rb') as f:
                files = {'files': ('test_document.txt', f, 'text/plain')}
                response = self.session.post(
                    f"{self.base_url}/sessions/{self.test_session_id}/upload",
                    files=files
                )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ File uploaded: {len(data.get('uploaded_files', []))} files")
                return True
            else:
                logger.error(f"❌ File upload failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ File upload error: {e}")
            return False
    
    def test_document_indexing(self) -> bool:
        """Test document indexing functionality"""
        logger.info("🔍 Testing document indexing...")
        try:
            response = self.session.post(f"{self.base_url}/sessions/{self.test_session_id}/index")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Documents indexed: {data.get('message', 'Success')}")
                return True
            else:
                logger.error(f"❌ Document indexing failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Document indexing error: {e}")
            return False
    
    def test_chat_functionality(self) -> bool:
        """Test chat functionality with different routing"""
        logger.info("🔍 Testing chat functionality...")
        try:
            # Test direct LLM chat (greeting)
            chat_data = {
                "message": "Hello, how are you?",
                "force_rag": False
            }
            response = self.session.post(
                f"{self.base_url}/sessions/{self.test_session_id}/messages",
                json=chat_data
            )
            
            if response.status_code == 200:
                data = response.json()
                used_rag = data.get('used_rag', False)
                logger.info(f"✅ Direct LLM chat: used_rag={used_rag}")
            else:
                logger.error(f"❌ Direct LLM chat failed: {response.status_code}")
                return False
            
            # Test RAG chat (document query)
            chat_data = {
                "message": "What does the document say about artificial intelligence?",
                "force_rag": True
            }
            response = self.session.post(
                f"{self.base_url}/sessions/{self.test_session_id}/messages",
                json=chat_data
            )
            
            if response.status_code == 200:
                data = response.json()
                used_rag = data.get('used_rag', False)
                source_docs = data.get('source_documents', [])
                logger.info(f"✅ RAG chat: used_rag={used_rag}, sources={len(source_docs)}")
                return True
            else:
                logger.error(f"❌ RAG chat failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Chat functionality error: {e}")
            return False
    
    def test_rag_endpoints(self) -> bool:
        """Test direct RAG endpoints"""
        logger.info("🔍 Testing direct RAG endpoints...")
        try:
            # Test RAG chat endpoint
            rag_data = {
                "query": "What information is available about machine learning?",
                "session_id": self.test_session_id,
                "force_rag": True
            }
            response = self.session.post(f"{self.base_url}/rag/chat", json=rag_data)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', '')
                sources = data.get('source_documents', [])
                logger.info(f"✅ RAG endpoint: answer_length={len(answer)}, sources={len(sources)}")
                return True
            else:
                logger.error(f"❌ RAG endpoint failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ RAG endpoint error: {e}")
            return False
    
    def test_index_management(self) -> bool:
        """Test index creation and management"""
        logger.info("🔍 Testing index management...")
        try:
            # Create index
            index_data = {
                "name": "Test Index",
                "description": "Test index for validation",
                "metadata": {"test": True}
            }
            response = self.session.post(f"{self.base_url}/indexes", json=index_data)
            
            if response.status_code == 201:
                data = response.json()
                self.test_index_id = data['index_id']
                logger.info(f"✅ Index created: {self.test_index_id}")
            else:
                logger.error(f"❌ Index creation failed: {response.status_code}")
                return False
            
            # List indexes
            response = self.session.get(f"{self.base_url}/indexes")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Indexes listed: {data.get('total', 0)} indexes")
                return True
            else:
                logger.error(f"❌ Index listing failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Index management error: {e}")
            return False
    
    def cleanup(self):
        """Clean up test resources"""
        logger.info("🧹 Cleaning up test resources...")
        try:
            # Delete test session
            if self.test_session_id:
                response = self.session.delete(f"{self.base_url}/sessions/{self.test_session_id}")
                if response.status_code == 200:
                    logger.info("✅ Test session deleted")
                else:
                    logger.warning(f"⚠️ Session deletion failed: {response.status_code}")
            
            # Delete test index
            if self.test_index_id:
                response = self.session.delete(f"{self.base_url}/indexes/{self.test_index_id}")
                if response.status_code == 200:
                    logger.info("✅ Test index deleted")
                else:
                    logger.warning(f"⚠️ Index deletion failed: {response.status_code}")
            
            # Delete test files
            for file_path in self.test_files:
                try:
                    os.unlink(file_path)
                    logger.info(f"✅ Test file deleted: {file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ File deletion failed: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def run_all_tests(self) -> bool:
        """Run all tests"""
        logger.info("🚀 Starting comprehensive RAG system tests...")
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Models Endpoint", self.test_models_endpoint),
            ("Session Management", self.test_session_management),
            ("File Upload", self.test_file_upload),
            ("Document Indexing", self.test_document_indexing),
            ("Chat Functionality", self.test_chat_functionality),
            ("RAG Endpoints", self.test_rag_endpoints),
            ("Index Management", self.test_index_management),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                if test_func():
                    passed += 1
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    failed += 1
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                failed += 1
                logger.error(f"❌ {test_name} ERROR: {e}")
            
            time.sleep(1)  # Brief pause between tests
        
        # Final results
        logger.info(f"\n{'='*50}")
        logger.info(f"TEST RESULTS")
        logger.info(f"{'='*50}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
        
        # Cleanup
        self.cleanup()
        
        return failed == 0

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the consolidated RAG system")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the server")
    parser.add_argument("--wait", action="store_true", help="Wait for server to be ready")
    parser.add_argument("--timeout", type=int, default=60, help="Server wait timeout")
    
    args = parser.parse_args()
    
    tester = RAGSystemTester(args.url)
    
    if args.wait:
        if not tester.wait_for_server(args.timeout):
            logger.error("❌ Server not ready, exiting")
            sys.exit(1)
    
    success = tester.run_all_tests()
    
    if success:
        logger.info("🎉 All tests passed! System is ready for deployment.")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed. Please check the logs and fix issues.")
        sys.exit(1)

if __name__ == "__main__":
    main() 