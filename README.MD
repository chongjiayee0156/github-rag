[watch demo](https://youtu.be/aV4cq7W9uY0)

```
Directory structure:
└── chongjiayee0156-github-rag/
    ├── README.MD
    ├── backend/
    │   ├── requirements.txt
    │   ├── .env.example
    │   └── app/
    │       ├── __init__.py
    │       ├── agent.py
    │       ├── main.py
    │       ├── schemas.py
    │       ├── state_manager.py
    │       └── tasks.py
    └── frontend/
        ├── index.html
        ├── script.js
        └── style.css

```

1. Set up the backend:
   - Install dependencies: `pip install -r requirements.txt`
   - Copy `.env.example` to `.env` and fill in your credentials. (Github PAT, Google Cloud Vertex AI Project ID, etc.)
   - Choose your preferred LLM model and text embedding model.
   - Run the FastAPI app: `uvicorn app.main:app --reload`
   - Access the API at `http://localhost:8000/docs`

2. Set up the frontend:
    - Open `index.html` in a web browser. (Avoid using `file://` protocol due to CORS issues, use a local server instead)
    - Include the local port and host in `backend/main.py` to allow CORS.
    - Use the form to input your query github repo. (only supports public repos)
    - Click "Submit" to send the request to the backend.
    - When the repo is done being processed, feel free to ask questions about the repo.
