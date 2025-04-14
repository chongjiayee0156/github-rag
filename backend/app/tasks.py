import logging
import uuid
from .agent import setup_graph_app, GraphState, _update_status, get_task_status # Added get_task_status import
from .state_manager import update_task_status # Corrected import (already had this)

logger = logging.getLogger(__name__)

# --- CORRECTED Function Definition ---
def run_repo_setup_task(task_id: str, repo_url: str, github_pat: str | None):
#                          ^^^^^^^^^^^ <-- Added task_id as the first parameter
    """The background task function to process a repository."""
    # task_id = str(uuid.uuid4()) # REMOVED: task_id is now passed in

    logger.info(f"Starting background task {task_id} for repo: {repo_url}")
    logger.info(f"Using Project ID: {os.getenv('GOOGLE_CLOUD_PROJECT_ID')}, Location: {os.getenv('GOOGLE_CLOUD_LOCATION')}") # Added logging

    # Initialize state for the graph
    initial_state = GraphState(
        task_id=task_id, # Use the passed-in task_id
        github_repo_url=repo_url,
        github_pat=github_pat,
        local_repo_path=None,
        documents=None,
        retriever=None,
        chroma_persist_dir=None,
        question=None,
        context=None,
        answer=None,
        error=None,
        current_step="Starting"
    )

    # Update initial status (ensure state_manager is imported correctly)
    _update_status(initial_state, "PROCESSING") # Use helper to log and update status

    try:
        logger.info(f"Task {task_id}: Invoking setup_graph_app...") # Added logging
        # Invoke the LangGraph setup application
        # This runs synchronously within the background task runner
        final_state = setup_graph_app.invoke(initial_state)
        logger.info(f"Task {task_id}: setup_graph_app invocation finished.") # Added logging


        # The graph itself should handle setting final status (COMPLETED/FAILED)
        # via the process_success_node or handle_error_node
        # Check the final status from the state manager as the graph might mutate it
        final_status_info = get_task_status(task_id)
        if final_status_info and final_status_info.get("status") == "FAILED":
             logger.error(f"Task {task_id} finished with FAILED status. Error: {final_status_info.get('error', 'Unknown - check graph logs')}")
        elif final_status_info and final_status_info.get("status") == "COMPLETED":
             logger.info(f"Task {task_id} finished successfully.")
        else:
             # Fallback check on the direct return state if status manager wasn't updated somehow
             if final_state.get("error"):
                 logger.error(f"Task {task_id} finished with error state in return value (status may be inconsistent): {final_state['error']}")
             else:
                 logger.warning(f"Task {task_id} finished, but final status in state manager is not COMPLETED or FAILED. Check graph logic. Final state status: {final_status_info.get('status') if final_status_info else 'Not Found'}")


    except Exception as e:
        # Catch unexpected errors during graph invocation itself
        logger.critical(f"Task {task_id}: Unhandled exception during graph execution for {repo_url}: {e}", exc_info=True)
        # Manually update status to FAILED if graph invocation crashed
        current_status = get_task_status(task_id) # Check status *before* updating
        # Only update if it hasn't already been marked completed/failed by the graph logic just before the crash
        if not current_status or current_status.get('status') not in ["COMPLETED", "FAILED"]:
             update_task_status(task_id, "FAILED", repo_url=repo_url, error=f"Unexpected graph execution error: {e}")
        # Attempt cleanup (might be redundant if graph handler worked)
        # from .agent import cleanup_task_state # Import locally if needed
        # cleanup_task_state(task_id) # Be careful about duplicate cleanup calls

    # No need to return task_id, it was generated and returned by the endpoint already
    # return task_id
# --- End Corrected Code ---

# Need to import os for getenv in logging line
import os
# Need get_task_status from agent.py
# Need correct state_manager import