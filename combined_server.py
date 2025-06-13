"""Combined runner that starts localGPT backend (port 8000)
   *and* Advanced RAG API (port 8001) in a single Python process.

This keeps existing code untouched—the script just imports the two `main()`
entry-points and launches them on separate threads.

Usage
-----
$ HF_TOKEN=... RAG_LOG_LEVEL=INFO python combined_server.py

Stop with Ctrl-C (both threads are daemons so the process exits cleanly).
"""
import os
import sys
import threading
import logging

# Ensure logging level picked up by children
os.environ.setdefault("RAG_LOG_LEVEL", os.getenv("RAG_LOG_LEVEL", "INFO"))

# Ensure repo root and backend directory are importable
REPO_ROOT = os.path.dirname(__file__)
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

# Import *after* modifying sys.path so local modules resolve
from backend import server as backend_server            # type: ignore
from rag_system import api_server as rag_api_server     # type: ignore


logger = logging.getLogger("combined_server")
logging.basicConfig(level=os.environ["RAG_LOG_LEVEL"].upper())


def _run_backend():
    logger.info("Starting legacy localGPT backend on port 8000 …")
    try:
        backend_server.main()
    except Exception as e:
        logger.error("Backend server crashed: %s", e, exc_info=True)


def _run_rag():
    logger.info("Starting Advanced RAG API on port 8001 …")
    try:
        if hasattr(rag_api_server, "start_server"):
            rag_api_server.start_server(port=8001)
        else:
            rag_api_server.main()  # type: ignore[attr-defined]
    except Exception as e:
        logger.error("RAG API server crashed: %s", e, exc_info=True)


if __name__ == "__main__":
    # Run backend on a daemon thread so Ctrl-C stops everything
    t_backend = threading.Thread(target=_run_backend, daemon=True, name="localGPT-thread")
    t_backend.start()

    # Run RAG on main thread (blocks until Ctrl-C)
    _run_rag() 