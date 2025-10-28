# RAG Chatbot

A local Retrieval-Augmented-Generation chatbot powered by Streamlit. The project downloads (or loads) your knowledge base and lets you chat with it through a web UI.

## Prerequisites

- **Python** 3.9 or newer
- **Git**
- (Windows) PowerShell or Command Prompt

## Installation

```powershell
# 1. Clone the repository
 git clone <YOUR-FORK-URL> rag-chatbot
 cd rag-chatbot

# 2. (Optional) create & activate a virtual environment
 python -m venv venv
 .\venv\Scripts\activate   # Windows

# 3. Run the app
 start.bat                    # or: python -m streamlit run app.py
 # This will automatically install dependencies and start the application
```

The Streamlit UI will open in your browser at <http://localhost:8501>.

## Usage

Once the web interface opens you can:

1. **Upload PDFs** via the sidebar "Upload PDF" button to add them to the knowledge base.
2. **Ask questions** in the chat box at the bottom of the page. The chatbot will answer using the uploaded documents.
3. Toggle **Show sources** to display which document chunks contributed to the answer.
4. Use **Rebuild Knowledge Base** to re-index all PDFs in `knowledge_base/`.
5. Use **Clear Knowledge Base** or **Clear History** at any time to start fresh.

## Project Layout

### Core Files
- `app.py` - Main Streamlit application with the web interface
- `config.py` - Configuration settings for the application
- `rag_pipeline.py` - Main RAG (Retrieval-Augmented Generation) pipeline implementation
- `llm_handler.py` - Handles loading and interacting with the language model
- `pdf_loader.py` - Processes and chunks PDF documents
- `vector_store.py` - Manages the vector database for document storage and retrieval

### Supporting Files
- `requirements.txt` - Python package dependencies
- `start.bat` - Windows script to install dependencies and launch the app
- `.gitignore` - Specifies intentionally untracked files to ignore

### Directories
- `knowledge_base/` - Stores uploaded PDF files (created on first run)
- `vector_store/` - Stores the vector database (created on first run)

## Contributing / Version control

```powershell
# First-time setup inside the project folder
 git init                   # create local repo
 git add .
 git commit -m "Initial commit"

# After making changes
 git add <files>
 git commit -m "Describe your change"
```

Push to GitHub/GitLab and share the repo link so collaborators can clone and run the same code.

---
Feel free to tweak the instructions or Python version numbers to match your environment.
