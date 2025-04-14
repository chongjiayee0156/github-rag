from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class RepoSetupRequest(BaseModel):
    github_repo_url: str
    github_pat: Optional[str] = None

class RepoSetupResponse(BaseModel):
    task_id: str
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str # e.g., "PENDING", "PROCESSING", "COMPLETED", "FAILED"
    error: Optional[str] = None
    repo_url: Optional[str] = None # Add repo url for context

class QuestionRequest(BaseModel):
    task_id: str # Use task_id to identify the correct retriever
    question: str

class AnswerResponse(BaseModel):
    answer: str
    error: Optional[str] = None

class GraphStateUpdate(BaseModel):
    """Model for sending graph state updates via SSE"""
    step: str
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None