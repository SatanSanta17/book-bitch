# Book Bitch (RAG Demo)

Weekend-ready Retrieval-Augmented Generation project:
1. Upload textbook PDF
2. Extract + chunk text
3. Embed + store in Pinecone/FAISS
4. Ask questions from Streamlit chat, grounded in the book

## Quickstart

```bash
cp .env.example .env
# Fill in keys before running: OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV

pip install -e .
make run-backend
make run-frontend
```

Visit http://localhost:8501 for UI and http://localhost:8000/docs for API docs.

### Tests

```bash
make test
```

### Safety & Copyright

Use public-domain or licensed PDFs. Delete uploads after demo. The system prompt enforces grounded answers only.