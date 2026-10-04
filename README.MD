# Intelligent Multi-Document Q&A System using LLM and RAG

An AI-powered document question-answering system that allows users
to upload multiple PDF files and ask questions based on their content.

## Features

- Upload multiple PDF documents
- Extract text from PDF files
- Split text into smaller chunks
- Generate embeddings using Sentence Transformers
- Retrieve relevant information using FAISS
- Generate answers using Groq LLM
- Display source filenames and page numbers
- Interactive interface built using Streamlit

## Technologies Used

- Python
- Streamlit
- Retrieval-Augmented Generation (RAG)
- FAISS
- Sentence Transformers
- Groq API
- PyPDF

## How It Works

1. Upload PDF documents.
2. Extract and split the document text into chunks.
3. Generate vector embeddings.
4. Store the embeddings in a FAISS index.
5. Convert the user's question into an embedding.
6. Retrieve the most relevant chunks.
7. Pass the retrieved context to the Groq LLM.
8. Display the generated answer with source references.

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL