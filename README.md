
# Semantic File Explorer

It is a desktop application that makes searching through files smarter by understanding the **meaning of your query**, not just matching keywords.

Instead of typing the exact file name or words inside a document, you can describe what you’re looking for in natural language (for example: *“machine learning optimization techniques”*), and LucidFiles will find the most relevant files based on semantic similarity.

The system combines **transformer-based sentence embeddings, vector search, and AI summarization** to create a modern file discovery experience.

---

# Motivation

Most file explorers like Finder, Windows Explorer, or Linux file managers rely on **keyword or filename matching**.

This causes problems such as:

* You forget the exact filename.
* The document contains related concepts but not the exact words.
* Searching across many formats (PDF, DOCX, images) is difficult.

LucidFiles solves this by using **Natural Language Processing (NLP)** to understand the **semantic intent** of a query and match it with document content.

---

# Key Features

### Semantic Search

Search files using natural language queries instead of exact keywords.

Example:

```
Query: "authentication security patterns"
```

LucidFiles may retrieve files like:

* `jwt_authentication_design.pdf`
* `secure_login_implementation.md`

even if the exact phrase does not exist.

---

### Multi-Format Document Support

LucidFiles can index content from many file formats:

* TXT / Markdown
* PDF
* DOCX
* Images (PNG, JPG, TIFF) using OCR
* Source code files

---

### AI Summarization

LucidFiles integrates **AI summarization** to generate quick summaries of documents.

You can:

* Click **Summarize**
* Get a concise explanation of the document
* Ask follow-up questions

---

### Real-Time File Indexing

The system automatically updates the search index whenever files change.

It detects:

* new files
* modified files
* deleted files

This means you never need to manually rebuild the index.

---

### Audio Podcast Generation

Documents can be converted into **audio summaries** using text-to-speech.

This allows users to **listen to documents like podcasts**.

---

# System Architecture

LucidFiles uses a multi-service architecture:

```
Frontend (React + Electron)
        │
        ▼
Backend (Node.js + Express)
        │
        ▼
Worker (Python + FastAPI)
        │
        ▼
Vector Database (Qdrant)
```

### Frontend

Desktop UI built with:

* React
* Electron

Provides:

* semantic search interface
* document preview
* AI insights
* podcast player

---

### Backend

Node.js server responsible for:

* API routing
* directory management
* metadata storage (SQLite)
* file watching

---

### Worker Service

Python FastAPI service responsible for:

* document parsing
* text chunking
* embedding generation
* semantic search

---

### Vector Database

LucidFiles uses **Qdrant** for storing embeddings and performing fast similarity search.

Embedding model used:

```
sentence-transformers/all-MiniLM-L6-v2
```

Vector size: **384 dimensions**

Search metric: **cosine similarity**

---

# How Search Works

The semantic search pipeline works as follows:

1. Documents are parsed into plain text.
2. Text is split into chunks.
3. Each chunk is converted into an embedding vector.
4. Vectors are stored in Qdrant.
5. When a user searches:

   * the query is embedded
   * similarity search retrieves the closest vectors
6. Results are ranked and returned to the UI.

---

# Tech Stack

Frontend

* React
* Electron
* TailwindCSS

Backend

* Node.js
* Express
* SQLite

Worker

* Python
* FastAPI
* SentenceTransformers

Database

* Qdrant Vector Database (Docker)

AI Services

* OpenAI GPT for summarization
* Tesseract OCR for images
* Text-to-Speech for podcast generation

---

# Installation

Clone the repository

```bash
git clone https://github.com/raghavvag/lucidfiles.git
cd lucidfiles
```

---

### Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

### Install backend dependencies

```bash
cd backend
npm install
npm run dev
```

---

### Start Python worker

```bash
cd worker
pip install -r requirements.txt
uvicorn main:app --reload
```

---

### Run the desktop app

```bash
cd frontend
npm install
npm run electron
```

---

# Example Query

```
Query:
"deep learning optimization methods"
```

Possible results:

* `adam_optimizer_notes.pdf`
* `gradient_descent_tutorial.md`
* `training_neural_networks.docx`

---

# Performance

Typical results from experiments:

| Metric                   | Value  |
| ------------------------ | ------ |
| Mean cosine similarity   | 0.82   |
| Precision@5              | 0.78   |
| Search latency (cached)  | < 30ms |
| Embedding cache hit rate | ~87%   |

---

# Limitations

* OCR accuracy is limited for handwritten documents.
* AI summarization requires an OpenAI API key.
* Initial embedding model loading may take a few seconds.

---

# Future Improvements

Planned improvements include:

* hybrid search (BM25 + embeddings)
* support for multimodal embeddings
* local LLM summarization
* distributed search across devices

---

# Contributors

* Kriti Maheshwari
* Raghav Agrawal
* Pranjal Agarwal

---

# License

This project is intended for academic and research purposes.
---
