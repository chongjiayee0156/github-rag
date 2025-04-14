const setupForm = document.getElementById('setup-form');
const setupStatusDiv = document.getElementById('setup-status');
const currentTaskIdInput = document.getElementById('current-task-id');

const qaSection = document.getElementById('qa-section');
const qaForm = document.getElementById('qa-form');
const qaResponseDiv = document.getElementById('qa-response');
const answerOutputPre = document.getElementById('answer-output');
const repoContextSpan = document.getElementById('repo-context');
const qaLoadingDiv = document.getElementById('qa-loading');

// --- Configuration ---
// Adjust if your backend runs on a different port/host
const API_BASE_URL = 'http://localhost:8000';
// --- End Configuration ---

let currentTaskPollingInterval = null; // To store the interval ID for status polling
let currentEventSource = null; // To store EventSource connection


function displaySetupStatus(message, statusClass = '') {
    setupStatusDiv.textContent = message;
    setupStatusDiv.className = statusClass; // 'processing', 'completed', 'failed'
}

function displayAnswer(message, isError = false) {
    answerOutputPre.textContent = message;
    answerOutputPre.className = isError ? 'error' : '';
}

function showQaSection(taskId, repoUrl) {
    currentTaskIdInput.value = taskId;
    repoContextSpan.textContent = repoUrl || taskId; // Show repo URL if available
    qaSection.style.display = 'block';
    displaySetupStatus(`Setup complete for ${repoUrl || taskId}. You can now ask questions.`, 'completed');
}

function hideQaSection() {
    qaSection.style.display = 'none';
    currentTaskIdInput.value = '';
    repoContextSpan.textContent = '';
    displayAnswer('Ask a question above...'); // Reset answer area
}

// --- Event Listener for Setup Form ---
setupForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    console.log("Submit prevented. Does page still reload?");
    hideQaSection(); // Hide Q&A while setting up new repo
    stopStatusUpdates(); // Stop previous polling/SSE

    const repoUrl = document.getElementById('repo-url').value;
    const githubPat = document.getElementById('github-pat').value;

    console.log(`Starting setup for repo: ${repoUrl} with PAT: ${githubPat ? 'Provided' : 'Not Provided'}`);

    displaySetupStatus('Starting repository setup...', 'processing');

    console.log(JSON.stringify({
        github_repo_url: repoUrl,
        github_pat: githubPat || null, // Send null if empty
    }),)

    try {
        const response = await fetch(`${API_BASE_URL}/api/setup-repo`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                github_repo_url: repoUrl,
                github_pat: githubPat || null, // Send null if empty
            }),
        });

        console.log("Setup response:", response);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: "Unknown server error" }));
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.detail}`);
        }

        const data = await response.json();
        const taskId = data.task_id;
        currentTaskIdInput.value = taskId; // Store task ID

        console.log(`Setup initiated with Task ID: ${taskId}`);

        displaySetupStatus(`Setup initiated (Task ID: ${taskId}). Waiting for completion...`, 'processing');


        // Start polling for status OR use Server-Sent Events
        startStatusUpdates(taskId); // Use SSE preferable

    } catch (error) {
        console.error('Error starting setup:', error);
        displaySetupStatus(`Error starting setup: ${error.message}`, 'failed');
        hideQaSection();
    }
});


// --- Function to Start Status Updates (Polling or SSE) ---
function startStatusUpdates(taskId) {
    console.log(`Starting status updates for task ${taskId}`);
    // Prioritize Server-Sent Events if available
    if (typeof(EventSource) !== "undefined") {
        console.log(`Starting SSE connection for task ${taskId}`);
        startSSEStatusUpdates(taskId);
    } else {
        // Fallback to polling
        console.log(`Starting polling for task ${taskId}`);
        startPollingStatusUpdates(taskId);
    }
}

// --- Function to Stop Status Updates ---
function stopStatusUpdates() {
    if (currentTaskPollingInterval) {
        clearInterval(currentTaskPollingInterval);
        currentTaskPollingInterval = null;
        console.log("Stopped status polling.");
    }
     if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
        console.log("Closed SSE connection.");
    }
}

// --- Status Updates using Server-Sent Events (Preferred) ---
function startSSEStatusUpdates(taskId) {
    stopStatusUpdates(); // Ensure only one update mechanism runs

    const eventSourceUrl = `${API_BASE_URL}/api/stream-status/${taskId}`;
    currentEventSource = new EventSource(eventSourceUrl);

    currentEventSource.addEventListener('status', function(event) {
        console.log("SSE status received:", event.data);
        const statusData = JSON.parse(event.data);
        handleStatusUpdate(statusData);
    });
    // Inside script.js -> startSSEStatusUpdates
    currentEventSource.onerror = function(errEvent) { // Changed name to errEvent
    // Log the full event object to see if it contains more details
    console.error("EventSource failed. Full event object:", errEvent);
    displaySetupStatus(`Error receiving status updates (SSE connection failed). Please check backend logs and console.`, 'failed');
    stopStatusUpdates();
};
}


// --- Status Updates using Polling (Fallback) ---
function startPollingStatusUpdates(taskId) {
    stopStatusUpdates(); // Ensure only one update mechanism runs

    currentTaskPollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/setup-status/${taskId}`);
            if (!response.ok) {
                // Handle 404 Not Found specifically - task might be unknown
                if(response.status === 404){
                    console.warn(`Polling: Task ${taskId} not found. Stopping polling.`);
                    displaySetupStatus(`Task ${taskId} not found. Setup might have failed early or ID is incorrect.`, 'failed');
                    stopStatusUpdates();
                    hideQaSection();
                    return; // Stop this interval check
                }
                // Other errors
                 const errorData = await response.json().catch(() => ({ detail: "Unknown server error" }));
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.detail}`);
            }
            const statusData = await response.json();
            handleStatusUpdate(statusData);

        } catch (error) {
            console.error('Error polling status:', error);
            displaySetupStatus(`Error polling status: ${error.message}. Stopping updates.`, 'failed');
            stopStatusUpdates(); // Stop polling on error
            hideQaSection();
        }
    }, 3000); // Poll every 3 seconds
}

// --- Common Handler for Status Data (from Polling or SSE) ---
function handleStatusUpdate(statusData) {
     console.log("Handling status update:", statusData);
     const { task_id, status, error, repo_url } = statusData;

    if (status === 'COMPLETED') {
        displaySetupStatus(`Setup complete for ${repo_url || task_id}.`, 'completed');
        stopStatusUpdates(); // Stop polling/SSE on completion
        showQaSection(task_id, repo_url);
    } else if (status === 'FAILED') {
        displaySetupStatus(`Setup failed for ${repo_url || task_id}: ${error || 'Unknown reason'}`, 'failed');
        stopStatusUpdates(); // Stop polling/SSE on failure
        hideQaSection();
    } else if (status === 'PROCESSING' || status === 'PENDING') {
        // Update status message but keep polling/SSE active
        displaySetupStatus(`Status for ${repo_url || task_id}: ${status}...`, 'processing');
    } else if (status === 'NOT_FOUND') {
         displaySetupStatus(`Task ${task_id} not found by server. Setup might have failed early or ID is incorrect.`, 'failed');
         stopStatusUpdates();
         hideQaSection();
    }
     else {
        // Keep showing processing for unknown or intermediate states
        displaySetupStatus(`Current status for ${repo_url || task_id}: ${status}...`, 'processing');
    }
}


// --- Event Listener for QA Form ---
qaForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    const question = document.getElementById('question').value;
    const taskId = currentTaskIdInput.value;

    if (!taskId) {
        displayAnswer("Error: No active repository setup found. Please setup a repository first.", true);
        return;
    }

    displayAnswer(''); // Clear previous answer
    qaLoadingDiv.style.display = 'block'; // Show thinking indicator

    try {
        const response = await fetch(`${API_BASE_URL}/api/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                task_id: taskId,
                question: question,
            }),
        });

         qaLoadingDiv.style.display = 'none'; // Hide thinking indicator

        if (!response.ok) {
             const errorData = await response.json().catch(() => ({ detail: "Unknown server error during QA" }));
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.detail}`);
        }

        const data = await response.json();
        displayAnswer(data.answer);

    } catch (error) {
        console.error('Error asking question:', error);
        qaLoadingDiv.style.display = 'none'; // Hide thinking indicator
        displayAnswer(`Error asking question: ${error.message}`, true);
    }
});