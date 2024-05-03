# Local Retrieval-Augmented Generation (RAG) Application

## Overview
Local retrieval-augmented generation app for querying text documents using NLP and retrieval techniques.

## How It Works
1. **Document Loading:** Loads `.txt` files, tokenizes documents.
2. **Query Embedding:** Embeds queries and documents using `HuggingFaceEmbeddings`.
3. **Retrieval System:** Manages embeddings with a SQLite database.
4. **Question Answering:** Uses `RetrievalQA` to generate answers.
5. **Output:** Displays answers.

## Installation
Requires Python 3.x, `nltk`, `spacy`, `langchain`, `langchain_community`. Install NLTK data and SpaCy `en_core_web_sm`.

## Usage
Run the script, input a question, and get an answer.

### Example
```bash
$ python local_rag.py
> What would you like to know about the transcripts?
> Who mentioned climate change?
"Several politicians mentioned climate change during the speeches..."
