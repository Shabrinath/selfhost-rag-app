# Selfhosted-RAG-App
Overview
--------

This is a simple internal **Retrieval-Augmented Generation (RAG)** application built using open-source tools. It allows you to ask natural language questions and get answers based on your own private documents (PDFs or text files).

The app uses the following stack:

-   **LangChain** for orchestration

-   **Chroma** for vector storage

-   **Ollama** for running local LLMs (like `llama3`)

-   **HuggingFace Sentence Transformers** for embedding text

-   **Streamlit** for the UI

* * * * *

Features
--------

-   Load and process your own PDF or text documents

-   Ask questions in natural language

-   Answers are generated using relevant chunks from your documents

-   Fully local: no cloud, no third-party APIs

* * * * *

How It Works
------------

1.  Documents are split into chunks.

2.  Chunks are embedded using HuggingFace models.

3.  Embeddings are stored in a local Chroma DB.

4.  At query time, relevant chunks are retrieved and passed to a local LLM via Ollama.

5.  LLM generates the final answer.

* * * * *

Setup Instructions
------------------

### 1\. Clone the Repository
```
git clone https://github.com/yourname/internal-rag.git
cd selfhosted-rag-app
```

### 2\. Create and Activate Virtual Environment
```
python3 -m venv rag-env
source rag-env/bin/activate
```

### 3\. Install Required Packages
```
pip install --upgrade pip
pip install -r requirements.txt
```

> Make sure the following extra packages are installed based on LangChain version warnings:
```
pip install langchain-community langchain-chroma langchain-huggingface langchain-ollama
```

### 4\. Install and Run Ollama

Install Ollama then pull and run the LLM you want (e.g., LLaMA 3):

```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
ollama run llama3
```
Keep this running in a separate terminal or background.

* * * * *

Folder Structure
----------------

```
selfhosted-rag-app/

├── app.py               # Main Streamlit app

├── ingest.py            # Script to process and load your documents

├── data/                # Folder where you put your PDFs or text files

├── db/                  # Chroma vector store folder (auto-created)

├── requirements.txt     # Python dependencies`

```
* * * * *

Running the App
---------------

### 1\. Start the App
```
streamlit run app.py
```

Then open your browser and go to:
```
http://localhost:8501
```
* * * * *

Notes
-----

-   You can use this on an EC2 instance as long as ports are open (`8501` by default).

-   All data stays local.

-   Works best with concise and structured documents.
