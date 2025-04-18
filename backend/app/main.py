import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse # Optional: For SSE status updates
import asyncio # Optional: For SSE
import os
import requests 
from fastapi.security import OAuth2PasswordBearer 
from typing import List, Optional 
from fastapi.responses import StreamingResponse
import time
from . import schemas, state_manager, tasks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GitHub RAG Agent API")

# Configure CORS (allow frontend to communicate)
origins = [
    "http://localhost",       # Allow local development
    "http://localhost:8000",  # Default frontend port if served separately
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500",
    "file:///Users/chongjiayee/Downloads/self-learn/saas-github-rag"
    # Add the URL of your deployed frontend if applicable
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/setup-repo", response_model=schemas.RepoSetupResponse)
async def setup_repository(
    request: schemas.RepoSetupRequest, background_tasks: BackgroundTasks
):
    """
    Endpoint to start the repository setup process in the background.
    """
    logger.info(f"Received setup request for repo: {request.github_repo_url}")

    # Add the setup task to run in the background
    # Note: BackgroundTasks in FastAPI run *after* the response is sent.
    # The task_id is generated inside the task function now.
    # We need a way to return it immediately. Let's pass a future or queue,
    # or simpler: generate task_id here and pass it to the task.

    # Let's adjust tasks.py and here:
    task_id = str(uuid.uuid4()) # Generate ID here
    
    # tasks.run_repo_setup_task(task_id, request.github_repo_url, request.github_pat)
    # ^-- For simplicity here, assume run_repo_setup_task *returns* the task_id
    # In a real scenario with true async execution before response, you'd need
    # a more complex mechanism (e.g., Celery task ID, or shared state).
    # Reverting run_repo_setup_task to accept task_id for this example:


    background_tasks.add_task(tasks.run_repo_setup_task, task_id, request.github_repo_url, request.github_pat)

    # Initial status before task actually starts running
    state_manager.update_task_status(task_id, "PENDING", repo_url=request.github_repo_url)

    logger.info(f"Task {task_id} queued for repo: {request.github_repo_url}")
    return schemas.RepoSetupResponse(task_id=task_id, message="Repository setup process started.")


@app.get("/api/setup-status/{task_id}", response_model=schemas.TaskStatusResponse)
async def get_setup_status(task_id: str):
    """
    Endpoint to check the status of a repository setup task.
    """
    status_info = state_manager.get_task_status(task_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Task not found")

    logger.debug(f"Status check for task {task_id}: {status_info}")
    return schemas.TaskStatusResponse(
        task_id=task_id,
        status=status_info.get("status", "UNKNOWN"),
        error=status_info.get("error"),
        repo_url=status_info.get("repo_url")
    )

# --- Optional: Server-Sent Events for Status Updates ---
# This requires the frontend JS to handle SSE connections
@app.get("/api/stream-status/{task_id}")
async def stream_setup_status(task_id: str, request: Request):
    """
    Streams status updates for a task using Server-Sent Events.
    """
    async def event_generator():
        last_status_sent = None
        logger.info(f"SSE Genrator starting loop for task {task_id}") # ADDED
        try: # ADDED try block
            while True:
                if await request.is_disconnected():
                    logger.info(f"SSE client disconnected for task {task_id}")
                    break

                # Log before getting status
                logger.debug(f"SSE Gen: Checking status for {task_id}")
                current_status_info = state_manager.get_task_status(task_id)
                logger.debug(f"SSE Gen: Got status for {task_id}: {current_status_info}") # Log retrieved status

                if not current_status_info:
                    # ... (keep existing logic)
                    yield {"event": "status", "data": json.dumps({"task_id": task_id, "status": "NOT_FOUND"})}
                    logger.warning(f"SSE: Task {task_id} not found during streaming.")
                    break

                # Log before creating JSON
                logger.debug(f"SSE Gen: Formatting status for {task_id}")
                # --- THIS IS THE INCORRECT LINE ---
                # current_status_json = json.dumps(schemas.TaskStatusResponse(
                # ).model_dump()) # or .dict()

                # --- REPLACE WITH THIS CORRECTED LINE ---
                current_status_json = json.dumps(schemas.TaskStatusResponse(
                    task_id=task_id, # Pass the task_id explicitly
                    status=current_status_info.get("status", "UNKNOWN"),
                    error=current_status_info.get("error"),
                    repo_url=current_status_info.get("repo_url")
                 ).model_dump()) # Use .dict() if on Pydantic v1.x
                # --- END OF CORRECTION ---
                logger.debug(f"SSE Gen: Formatted JSON for {task_id}: {current_status_json}")

                if current_status_json != last_status_sent:
                    logger.info(f"SSE Gen: Yielding update for {task_id}: {current_status_json}") # Log before yield
                    yield {"event": "status", "data": current_status_json} # This might raise error if client disconnected mid-yield
                    last_status_sent = current_status_json
                    logger.debug(f"SSE Gen: Yielded update for {task_id}") # Log after yield

                if current_status_info.get("status") in ["COMPLETED", "FAILED"]:
                    logger.info(f"SSE: Task {task_id} reached terminal state. Stopping stream.")
                    break

                logger.debug(f"SSE Gen: Sleeping for {task_id}") # Log before sleep
                await asyncio.sleep(2)
                logger.debug(f"SSE Gen: Woke up for {task_id}") # Log after sleep

        except asyncio.CancelledError:
            logger.info(f"SSE Generator for {task_id} cancelled (likely client disconnect).") # Handle expected cancellation
        except Exception as e:
            # !!! CATCH AND LOG UNEXPECTED ERRORS !!!
            logger.error(f"!!! SSE Generator ERROR for task {task_id}: {e}", exc_info=True)
            # Optionally yield an error message to the client before closing
            try:
                yield {"event": "error", "data": json.dumps({"message": "Internal SSE error occurred"})}
            except Exception:
                pass # Ignore error if yield fails after initial error
        finally:
            logger.info(f"SSE Generator loop finished for task {task_id}") # ADDED finally block
        
        
    # Need to import json for SSE data
    import json
    logger.info(f"SSE connection established for task {task_id}")
    return EventSourceResponse(event_generator())
# --- End Optional SSE ---


@app.post("/api/ask", response_model=schemas.AnswerResponse)
async def ask_question(request: schemas.QuestionRequest):
    """
    Endpoint to ask a question about a previously processed repository.
    """
    logger.info(f"Received question for task {request.task_id}: '{request.question}'")

    # 1. Check task status
    status_info = state_manager.get_task_status(request.task_id)
    if not status_info:
        raise HTTPException(status_code=404, detail=f"Task '{request.task_id}' not found.")
    if status_info.get("status") != "COMPLETED":
         raise HTTPException(status_code=400, detail=f"Repository setup for task '{request.task_id}' is not complete (status: {status_info.get('status', 'UNKNOWN')}). Please wait or check status.")

    # 2. Retrieve the retriever
    retriever = state_manager.get_retriever(request.task_id)
    if not retriever:
        # This shouldn't happen if status is COMPLETED, but check anyway
        logger.error(f"Task {request.task_id} status is COMPLETED but retriever not found in state manager.")
        raise HTTPException(status_code=500, detail=f"Internal error: Knowledge base for task '{request.task_id}' not available despite completed status.")

    # 3. Run the query (using the direct query function from agent.py)
    # This runs synchronously in the request thread. For very long queries, consider background tasks too.
    from .agent import query_repository # Import locally to avoid circular dependency issues at module level
    result = query_repository(retriever, request.question)

    if result.get("error"):
        # Log the error but return a user-friendly message
        logger.error(f"Error querying task {request.task_id}: {result['error']}")
        # Don't expose potentially sensitive internal error details directly
        # raise HTTPException(status_code=500, detail=f"Error generating answer: {result['error']}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the answer.")


    return schemas.AnswerResponse(answer=result["answer"])

@app.post("/api/search-repos", response_model=schemas.SearchResponse)
async def search_repositories(request: schemas.SearchRequest):
    """
    Endpoint to search repositories on GitHub using the Search API.
    """
    logger.info(f"Received search request with query: {request.query}")
    github_api_url = f"https://api.github.com/search/repositories"
    
    # Use PAT for authentication if provided (better rate limits)
    headers = {"Accept": "application/vnd.github.v3+json"}
    if request.github_pat:
        headers["Authorization"] = f"token {request.github_pat}"
        logger.info("Using provided GitHub PAT for search API.")
    else:
        logger.warning("No GitHub PAT provided for search, using unauthenticated request (lower rate limits).")
        
            
    # Construct query parameters - keep it simple for now
    # You can add more params like sort, order, per_page later
    params = {'q': request.query, 'per_page': 20} # Limit to 20 results for now
    
    logger.info(f"Calling GitHub Search API with params: {params}")

    try:
        response = requests.get(github_api_url, headers=headers, params=params)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        search_data = response.json()
        
        # Extract relevant info for the frontend
        results = []
        for item in search_data.get("items", []):
            if item.get("html_url") and item.get("name") and item.get("full_name") and item.get("owner"):
                 results.append(
                     schemas.RepositoryInfo(
                         name=item["name"],
                         full_name=item["full_name"],
                         html_url=item["html_url"],
                         description=item.get("description"),
                         owner_login=item["owner"]["login"]
                     )
                 )
            
        logger.info(f"Search returned {len(results)} repositories.")
        return schemas.SearchResponse(items=results)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling GitHub Search API: {e}", exc_info=True)
        # Check for specific status codes if needed
        status_code = e.response.status_code if e.response is not None else 500
        detail = f"Error contacting GitHub Search API: {e}"
        if status_code == 401:
             detail = "GitHub API authentication failed. Check your PAT."
        elif status_code == 403:
             detail = "GitHub API rate limit exceeded or access forbidden. Try adding/checking your PAT."
        elif status_code == 422:
             detail = "GitHub API validation failed. Check your search query syntax."
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e:
        logger.error(f"Unexpected error during repository search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during search.")
    
# Need uuid for task_id generation
import uuid

if __name__ == "__main__":
    # This is for running locally, typically you'd use uvicorn command line
    import uvicorn
    logger.info("Starting Uvicorn server...")
    # Ensure base Chroma directory exists
    os.makedirs(tasks.CHROMA_BASE_PERSIST_DIR, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)