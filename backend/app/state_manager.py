from typing import Dict, Any, Optional
from langchain_core.vectorstores import VectorStoreRetriever
import threading

# WARNING: This is simple in-memory state management.
# It will be lost if the server restarts.
# Not suitable for production - use Redis, DB, etc. for real SaaS.

# Use thread locks for basic safety if using threaded frameworks like default FastAPI/Uvicorn
_lock = threading.Lock()

# Stores the retriever associated with a completed task_id (or repo_url)
retrievers: Dict[str, VectorStoreRetriever] = {}

# Stores the status and potential errors of ongoing/completed tasks
task_statuses: Dict[str, Dict[str, Any]] = {}

def get_retriever(task_id: str) -> Optional[VectorStoreRetriever]:
    with _lock:
        return retrievers.get(task_id)

def store_retriever(task_id: str, retriever: VectorStoreRetriever):
    with _lock:
        retrievers[task_id] = retriever

def update_task_status(task_id: str, status: str, repo_url: Optional[str] = None, error: Optional[str] = None):
    with _lock:
        if task_id not in task_statuses:
            task_statuses[task_id] = {}
        task_statuses[task_id]['status'] = status
        if repo_url:
             task_statuses[task_id]['repo_url'] = repo_url
        if error:
            task_statuses[task_id]['error'] = error
        else:
             # Clear error if status is not FAILED
             if status != "FAILED" and 'error' in task_statuses[task_id]:
                 del task_statuses[task_id]['error']


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return task_statuses.get(task_id)

def cleanup_task_state(task_id: str):
     """Removes state associated with a task_id."""
     with _lock:
          retrievers.pop(task_id, None)
          task_statuses.pop(task_id, None)
          print(f"Cleaned up state for task_id: {task_id}")