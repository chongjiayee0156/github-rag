import os
import shutil
import tempfile
import logging
from typing import Dict, TypedDict, Optional, List, Any
from operator import itemgetter

from dotenv import load_dotenv
from git import Repo, GitCommandError
from langchain_community.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from langgraph.graph import StateGraph, END

# Import state manager functions
from .state_manager import store_retriever, update_task_status, get_task_status, cleanup_task_state

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Environment Setup ---
load_dotenv()

# --- Configuration ---
DEFAULT_GITHUB_PAT = os.getenv("GITHUB_PAT")
VERTEXAI_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
VERTEXAI_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
EMBEDDING_MODEL_NAME = "text-embedding-004"
CHAT_MODEL_NAME = "gemini-2.0-flash-001"
CHROMA_COLLECTION_NAME_PREFIX = "github_repo_"
# Use a directory within the backend for persistent Chroma storage per task
# A real SaaS would need a more robust central location or managed DB
CHROMA_BASE_PERSIST_DIR = "./chroma_db_store"


# --- State Definition for LangGraph ---
class GraphState(TypedDict):
    task_id: str # Added task_id to track context
    github_repo_url: str
    github_pat: Optional[str]
    local_repo_path: Optional[str]
    documents: Optional[List[Document]]
    retriever: Optional[VectorStoreRetriever]
    chroma_persist_dir: Optional[str] # Path for this task's chroma data
    question: Optional[str]
    context: Optional[List[Document]]
    answer: Optional[str]
    error: Optional[str]
    current_step: Optional[str] # Track current step for status updates


# --- Helper to Update Task Status ---
def _update_status(state: GraphState, status: str, error: Optional[str] = None):
    task_id = state.get("task_id")
    repo_url = state.get("github_repo_url")
    step = state.get("current_step", "Unknown Step")
    if task_id:
        details = f"Step: {step}"
        update_task_status(task_id, status, repo_url=repo_url, error=error or state.get("error"))
        logger.info(f"Task {task_id} status updated: {status} at step '{step}'" + (f" Error: {error}" if error else ""))
    else:
        logger.warning("task_id missing in state, cannot update status.")

# --- Modified Node Functions for LangGraph ---

def set_step(step_name: str):
    """Decorator or wrapper to set the current step in the state."""
    def decorator(func):
        def wrapper(state: GraphState) -> GraphState:
            state['current_step'] = step_name
            logger.info(f"Task {state.get('task_id', 'N/A')}: Entering step '{step_name}'")
            return func(state)
        return wrapper
    return decorator

@set_step("Validating Inputs")
def validate_inputs(state: GraphState) -> GraphState:
    if not state.get("github_repo_url"):
        return {**state, "error": "GitHub repository URL is required."}

    pat = state.get("github_pat") or DEFAULT_GITHUB_PAT
    if not VERTEXAI_PROJECT_ID or not VERTEXAI_LOCATION:
         return {**state, "error": "GOOGLE_CLOUD_PROJECT_ID and GOOGLE_CLOUD_LOCATION must be set in .env"}

    # Generate a unique directory for this task's Chroma data
    task_id = state["task_id"]
    persist_dir = os.path.join(CHROMA_BASE_PERSIST_DIR, task_id)
    os.makedirs(persist_dir, exist_ok=True)


    logger.info(f"Task {task_id}: Validated inputs for {state['github_repo_url']}. PAT used: {'Yes' if pat else 'No'}. Persist dir: {persist_dir}")
    return {**state, "github_pat": pat, "chroma_persist_dir": persist_dir, "error": None}

@set_step("Cloning Repository")
def clone_repository(state: GraphState) -> GraphState:
    repo_url = state["github_repo_url"]
    pat = state.get("github_pat")
    task_id = state["task_id"]
    temp_dir = tempfile.mkdtemp(prefix=f"repo_{task_id}_")

    try:
        repo_url_with_pat = repo_url
        if pat:
            pat_cleaned = pat.replace("https://", "")
            repo_url_with_pat = repo_url.replace("https://", f"https://{pat_cleaned}@")

        logger.info(f"Task {task_id}: Cloning {repo_url} into {temp_dir}...")
        Repo.clone_from(repo_url_with_pat, temp_dir)
        logger.info(f"Task {task_id}: Cloning successful.")
        return {**state, "local_repo_path": temp_dir, "error": None}
    except GitCommandError as e:
        logger.error(f"Task {task_id}: Error cloning repository: {e}", exc_info=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
        error_message = f"Failed to clone repository: {e}. Check URL, permissions, and PAT if private."
        return {**state, "local_repo_path": None, "error": error_message}
    except Exception as e:
        logger.error(f"Task {task_id}: Unexpected error during cloning: {e}", exc_info=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {**state, "local_repo_path": None, "error": f"An unexpected error occurred during cloning: {e}"}

@set_step("Loading Documents")
def load_documents_from_repo(state: GraphState) -> GraphState:
    local_path = state.get("local_repo_path")
    task_id = state["task_id"]
    if not local_path:
        # This case should ideally be caught by the conditional edge after cloning fails
        return {**state, "error": "Local repository path not found (previous step likely failed)."}

    try:
        repo = Repo(local_path)
        default_branch = repo.remotes.origin.refs.HEAD.ref.remote_head
        
        loader = GitLoader(
            repo_path=local_path,
            branch = default_branch,
            file_filter=lambda file_path: file_path.endswith((".py", ".md", ".txt", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".html", ".css", ".ipynb", "Dockerfile", "Makefile", ".sh", ".yaml", ".json")) # Expanded list
        )
        documents = loader.load()
        # Filter out very small/empty docs
        documents = [doc for doc in documents if len(doc.page_content.strip()) > 20]

        logger.info(f"Task {task_id}: Loaded {len(documents)} documents.")
        if not documents:
             logger.warning(f"Task {task_id}: No relevant documents loaded. Check repo content/filters.")
             # Decide if this is an error state or just an empty repo
             # return {**state, "documents": [], "error": "No relevant documents found."} # Option A: Treat as error
             return {**state, "documents": [], "error": None} # Option B: Allow empty repo

        return {**state, "documents": documents, "error": None}
    except Exception as e:
        logger.error(f"Task {task_id}: Error loading documents: {e}", exc_info=True)
        error_detail = f"Failed to load documents: {e}"
        if isinstance(e, GitCommandError) and "did not match any file(s) known to git" in str(e.stderr):
            error_detail += " Hint: Could not find the default branch."
        return {**state, "documents": None, "error": error_detail}

@set_step("Splitting Documents")
def split_documents(state: GraphState) -> GraphState:
    documents = state.get("documents")
    task_id = state["task_id"]
    if not documents:
        logger.info(f"Task {task_id}: No documents to split.")
        return {**state, "documents": []} # Ensure documents is an empty list

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Task {task_id}: Split {len(documents)} documents into {len(split_docs)} chunks.")
        # Replace original documents with split ones for embedding
        return {**state, "documents": split_docs, "error": None}
    except Exception as e:
        logger.error(f"Task {task_id}: Error splitting documents: {e}", exc_info=True)
        return {**state, "documents": None, "error": f"Failed to split documents: {e}"}


@set_step("Creating Vector Store")
def create_vector_store_and_retriever(state: GraphState) -> GraphState:
    documents = state.get("documents")
    task_id = state["task_id"]
    persist_dir = state.get("chroma_persist_dir")

    if not documents:
         logger.warning(f"Task {task_id}: No documents available to create vector store.")
         # If no docs were loaded, we can't create a retriever. Mark as error? Or allow empty retriever?
         # Let's treat it as an error for RAG purposes.
         return {**state, "retriever": None, "error": "No documents found to build the knowledge base."}
    if not persist_dir:
        return {**state, "retriever": None, "error": "ChromaDB persistence directory not set."}

    try:
        logger.info(f"Task {task_id}: Initializing Vertex AI Embeddings ({EMBEDDING_MODEL_NAME})...")
        embeddings = VertexAIEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            project=VERTEXAI_PROJECT_ID, # Keep these if they work, remove if relying on env vars
            location=VERTEXAI_LOCATION
        )

        collection_name = f"{CHROMA_COLLECTION_NAME_PREFIX}{task_id}"
        logger.info(f"Task {task_id}: Creating ChromaDB vector store (collection: {collection_name}) from {len(documents)} chunks in {persist_dir}...")

        # Use persistence
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir
        )

        # It's good practice to explicitly persist changes if needed, though from_documents often handles it.
        # vector_store.persist() # Might not be strictly necessary after from_documents

        retriever = vector_store.as_retriever(search_kwargs={'k': 5})
        logger.info(f"Task {task_id}: ChromaDB vector store and retriever created successfully.")

        # *** Store the retriever in the central state manager ***
        store_retriever(task_id, retriever)
        # *******************************************************

        return {**state, "retriever": retriever, "error": None} # Keep retriever in state for potential chaining within graph if needed

    except Exception as e:
        logger.error(f"Task {task_id}: Error creating/persisting ChromaDB vector store: {e}", exc_info=True)
        # Clean up potentially corrupted chroma dir?
        # shutil.rmtree(persist_dir, ignore_errors=True) # Be cautious with auto-cleanup
        return {**state, "retriever": None, "error": f"Failed to create vector store: {e}"}

@set_step("Cleaning Up Cloned Repo")
def cleanup_local_repo(state: GraphState) -> GraphState:
    local_path = state.get("local_repo_path")
    task_id = state["task_id"]
    if local_path and os.path.exists(local_path):
        try:
            shutil.rmtree(local_path)
            logger.info(f"Task {task_id}: Removed temporary repo directory: {local_path}")
            return {**state, "local_repo_path": None} # Clear the path from state
        except Exception as e:
            logger.warning(f"Task {task_id}: Failed to remove temporary directory {local_path}: {e}")
            # Don't set an error here, just log warning
            return {**state}
    return {**state}


@set_step("Handling Error")
def handle_error(state: GraphState) -> GraphState:
    """Node to log the error and ensure cleanup happens."""
    error = state.get("error", "Unknown error")
    task_id = state.get("task_id", "N/A")
    logger.error(f"Task {task_id}: Workflow failed at step '{state.get('current_step', 'N/A')}'. Error: {error}")
    _update_status(state, "FAILED", error) # Update global status
    # We might have already cleaned the repo if the error happened after clone
    # but this ensures cleanup happens if error was before repo cleanup node
    final_state = cleanup_local_repo(state)
    # Also clean up potentially created chroma directory on failure
    persist_dir = final_state.get("chroma_persist_dir")
    if persist_dir and os.path.exists(persist_dir):
         try:
              shutil.rmtree(persist_dir)
              logger.info(f"Task {task_id}: Cleaned up Chroma directory on error: {persist_dir}")
         except Exception as e:
              logger.warning(f"Task {task_id}: Failed to clean up Chroma directory {persist_dir} on error: {e}")

    # Clean up retriever/task status from memory manager as well
    if task_id != "N/A":
         cleanup_task_state(task_id)

    return {**final_state, "error": error} # Propagate error state

@set_step("Setup Complete")
def process_success(state: GraphState) -> GraphState:
    """Node to mark successful completion."""
    task_id = state["task_id"]
    logger.info(f"Task {task_id}: Repository setup workflow completed successfully.")
    _update_status(state, "COMPLETED")
    # Retriever should already be stored by create_vector_store_and_retriever
    # Clean up the local repo clone directory if it wasn't done yet
    final_state = cleanup_local_repo(state)
    return {**final_state}


# --- Conditional Edge Logic ---
def should_continue(state: GraphState) -> str:
    if state.get("error"):
        # Log the error transition decision
        logger.warning(f"Task {state.get('task_id', 'N/A')}: Error detected in step '{state.get('current_step', 'N/A')}', routing to error handler. Error: {state['error']}")
        return "handle_error_node" # Route to the central error handler node
    else:
        return "continue"

# --- Build the Setup Graph ---
def compile_setup_graph():
    workflow_setup = StateGraph(GraphState)

    # Add nodes using the step-setting wrappers
    workflow_setup.add_node("validate_inputs", validate_inputs)
    workflow_setup.add_node("clone_repository", clone_repository)
    workflow_setup.add_node("load_documents", load_documents_from_repo)
    workflow_setup.add_node("split_documents", split_documents)
    workflow_setup.add_node("create_retriever", create_vector_store_and_retriever)
    # workflow_setup.add_node("cleanup_repo_node", cleanup_local_repo) # Cleanup now happens in success/error nodes
    workflow_setup.add_node("handle_error_node", handle_error)
    workflow_setup.add_node("process_success_node", process_success)

    # Define edges
    workflow_setup.set_entry_point("validate_inputs")

    # Conditional edges routing to 'handle_error_node' on failure
    workflow_setup.add_conditional_edges("validate_inputs", should_continue, {"continue": "clone_repository", "handle_error_node": "handle_error_node"})
    workflow_setup.add_conditional_edges("clone_repository", should_continue, {"continue": "load_documents", "handle_error_node": "handle_error_node"})
    workflow_setup.add_conditional_edges("load_documents", should_continue, {"continue": "split_documents", "handle_error_node": "handle_error_node"})
    workflow_setup.add_conditional_edges("split_documents", should_continue, {"continue": "create_retriever", "handle_error_node": "handle_error_node"})
    # After retriever creation (the last major step that populates the key resource)
    workflow_setup.add_conditional_edges("create_retriever", should_continue, {"continue": "process_success_node", "handle_error_node": "handle_error_node"})

    # End nodes
    workflow_setup.add_edge("process_success_node", END)
    workflow_setup.add_edge("handle_error_node", END) # Error handler is also a terminal node

    return workflow_setup.compile()

# --- Query Logic (Simpler: Direct function call, no separate graph needed for this part) ---
def query_repository(retriever: VectorStoreRetriever, question: str) -> Dict[str, Any]:
    """Generates an answer using the LLM with the retrieved context."""
    logger.info(f"Querying repository with question: '{question}'")

    try:
        logger.info(f"Initializing ChatVertexAI model ({CHAT_MODEL_NAME})...")
        llm = ChatVertexAI(
            model_name=CHAT_MODEL_NAME,
            project=VERTEXAI_PROJECT_ID, # Keep or remove based on previous results
            location=VERTEXAI_LOCATION,
            temperature=0.1, # Lower temperature for more factual RAG
        )

        rag_template = """ Your are a helpful assistant for studying a specific repository.
        Based solely on the following context from the codebase, answer the question.
        
        1. If the question is about a specific code snippet, provide a detailed explanation of that code.
        2. If the question is about a tutorial of the codebase, provide an easy-to-understand tutorial chapter.
            - start off by explaining to 5 year old what the code does.
            - then, slowly explain the code in a step-by-step manner.
            - use code snippets to illustrate the explanation.
            - use analogies to help the reader understand.
            - use examples to clarify the concepts.
            - use diagrams to illustrate complex concepts.
        3. If the question is about writing a README, provide a beginner-friendly README section
            - Include a title.
            - Include a beginner friendly description
            - Include usage instructions.
            - Include tech stack.
            - Output in Markdown format.
        


Context:
{context}

Question: {question}
Answer:"""
        prompt = ChatPromptTemplate.from_template(rag_template)

        def format_docs(docs: List[Document]):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("Invoking RAG chain...")
        answer = rag_chain.invoke(question)
        logger.info("Answer generated successfully.")
        return {"answer": answer, "error": None}

    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        return {"answer": None, "error": f"Failed to generate answer using LLM: {e}"}

# Compile the graph when the module is loaded
setup_graph_app = compile_setup_graph()