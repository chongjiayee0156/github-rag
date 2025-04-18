// --- Get new/changed elements ---
const searchForm = document.getElementById('search-form'); // Changed from setupForm
const searchStatusDiv = document.getElementById('search-status');
const searchResultsDiv = document.getElementById('search-results');
const resultsListUl = document.getElementById('results-list');
const setupStatusDiv = document.getElementById('setup-status'); // Keep this for setup phase
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

// --- Helper Functions ---
function displaySearchStatus(message, isError = false) {
    searchStatusDiv.textContent = message;
    searchStatusDiv.className = `status-message ${isError ? 'failed' : 'processing'}`;
    searchStatusDiv.style.display = 'block';
}

function clearSearchResults() {
    resultsListUl.innerHTML = '';
    searchResultsDiv.style.display = 'none'; // Hide results section
    searchStatusDiv.style.display = 'none'; // Hide search status
}


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

searchForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    console.log("Search form submitted.");

    // Clear previous results and statuses
    clearSearchResults();
    displaySetupStatus(''); // Clear setup status
    hideQaSection();
    stopStatusUpdates(); // Stop any previous SSE/polling

    const searchQuery = document.getElementById('search-query').value;
    const githubPat = document.getElementById('github-pat').value;

    displaySearchStatus('Searching repositories on GitHub...', false);
    searchResultsDiv.style.display = 'block'; // Show results section (initially empty list)

    try {
        const response = await fetch(`${API_BASE_URL}/api/search-repos`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: searchQuery,
                github_pat: githubPat || null
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: "Unknown server error during search" }));
            throw new Error(`Search failed: ${response.status} - ${errorData.detail}`);
        }

        const data = await response.json();

        if (!data.items || data.items.length === 0) {
            displaySearchStatus('No repositories found matching your query.', false);
             resultsListUl.innerHTML = '<li>No results found.</li>';
            return;
        }

        displaySearchStatus(`Found ${data.items.length} repositories. Select one to analyze.`, false);

        // Populate results list
        resultsListUl.innerHTML = ''; // Clear previous results
        data.items.forEach(repo => {
            const li = document.createElement('li');
            li.innerHTML = `
                <strong>${repo.full_name}</strong> (${repo.owner_login})<br>
                <small>${repo.description || 'No description'}</small><br>
                <button class="analyze-button" data-repo-url="${repo.html_url}">Analyze This Repo</button>
            `;
            // Add event listener to the button INSIDE this loop
            li.querySelector('.analyze-button').addEventListener('click', handleAnalyzeButtonClick);
            resultsListUl.appendChild(li);
        });

    } catch (error) {
        console.error('Error searching repositories:', error);
        displaySearchStatus(`Error searching: ${error.message}`, true);
    }
});

// --- NEW Handler for "Analyze This Repo" button clicks ---
async function handleAnalyzeButtonClick(event) {
    const repoUrl = event.target.getAttribute('data-repo-url');
    const githubPat = document.getElementById('github-pat').value; // Get PAT again

    if (!repoUrl) return;

    console.log(`Analyze button clicked for repo: ${repoUrl}`);
    
    // --- Now trigger the ORIGINAL setup process ---
    displaySetupStatus(''); // Clear previous setup message
    hideQaSection();
    stopStatusUpdates();

    // Visually indicate which repo is being processed (optional)
     resultsListUl.querySelectorAll('.analyze-button').forEach(btn => btn.disabled = true); // Disable all buttons
     event.target.textContent = 'Analyzing...'; // Change text of clicked button
     // change color of button to indicate processing
        event.target.style.backgroundColor = '#ccc'; // Change color to indicate processing


    displaySetupStatus(`Starting analysis for ${repoUrl}...`, 'processing');

    try {
         // --- Call the ORIGINAL /api/setup-repo endpoint ---
        const response = await fetch(`${API_BASE_URL}/api/setup-repo`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                github_repo_url: repoUrl,
                github_pat: githubPat || null,
            }),
        });

        // Re-enable buttons after call (even if failed)
         resultsListUl.querySelectorAll('.analyze-button').forEach(btn => {
            btn.disabled = false;
            if(btn === event.target) btn.textContent = 'Analyze This Repo'; // Reset text
         });


        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: "Unknown server error starting setup" }));
            throw new Error(`Setup failed: ${response.status} - ${errorData.detail}`);
        }

        const data = await response.json();
        const taskId = data.task_id;
        currentTaskIdInput.value = taskId; // Store task ID

        console.log(`Setup initiated with Task ID: ${taskId} for repo ${repoUrl}`);
        displaySetupStatus(`Setup initiated (Task ID: ${taskId}) for ${repoUrl}. Waiting for completion...`, 'processing');

        // Start polling/SSE for status updates (using the existing functions)
        startStatusUpdates(taskId, repoUrl); // Pass repoUrl for context

    } catch (error) {
        console.error('Error starting setup:', error);
        displaySetupStatus(`Error starting setup for ${repoUrl}: ${error.message}`, 'failed');
        hideQaSection();
         // Re-enable buttons if error occurred before fetch completed fully
         resultsListUl.querySelectorAll('.analyze-button').forEach(btn => {
            btn.disabled = false;
             if(btn === event.target) btn.textContent = 'Analyze This Repo';
         });
    }
}


// --- Status Update Functions (startStatusUpdates, stopStatusUpdates, startSSE, startPolling, handleStatusUpdate) ---
// MODIFY handleStatusUpdate slightly to use the repoUrl passed in startStatusUpdates
let currentRepoUrlForStatus = null; // Store repoUrl context for status messages

function startStatusUpdates(taskId, repoUrl = null) { // Accept repoUrl
    console.log(`Starting status updates for task ${taskId} (Repo: ${repoUrl})`);
    currentRepoUrlForStatus = repoUrl; // Store it
    if (typeof(EventSource) !== "undefined") {
        startSSEStatusUpdates(taskId);
    } else {
        startPollingStatusUpdates(taskId);
    }
}

// Modify handleStatusUpdate to use currentRepoUrlForStatus
function handleStatusUpdate(statusData) {
    console.log("Handling status update:", statusData);
    // If statusData itself includes repo_url, use that, otherwise use the stored one
    const repoUrl = statusData.repo_url || currentRepoUrlForStatus || statusData.task_id;
    const { task_id, status, error } = statusData;

    // Logic remains mostly the same, just uses the resolved repoUrl for display
    if (status === 'COMPLETED') {
        displaySetupStatus(`Setup complete for ${repoUrl}.`, 'completed');
        stopStatusUpdates();
        showQaSection(task_id, repoUrl); // Show QA section
    } else if (status === 'FAILED') {
        displaySetupStatus(`Setup failed for ${repoUrl}: ${error || 'Unknown reason'}`, 'failed');
        stopStatusUpdates();
        hideQaSection();
    } else if (status === 'PROCESSING' || status === 'PENDING') {
        displaySetupStatus(`Status for ${repoUrl}: ${status}...`, 'processing');
    } else if (status === 'NOT_FOUND') {
         displaySetupStatus(`Task ${task_id} not found by server. Setup might have failed early or ID is incorrect.`, 'failed');
         stopStatusUpdates();
         hideQaSection();
    }
     else {
        displaySetupStatus(`Current status for ${repoUrl}: ${status}...`, 'processing');
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