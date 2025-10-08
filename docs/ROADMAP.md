# Roadmap

## Current Stage
- ✅ Repo initialized
- ✅ Core architecture defined (local-first RAG, Weaviate + MiniLM + llama.cpp)
- 🟡 Next: ingestion + hybrid search implementation

## Upcoming Milestones
1. **Data Ingestion**
   - Parse and clean *Prose Edda* and *Bibliotheca*.
   - Convert Kaggle CSV into graph format.

2. **Embedding & Indexing**
   - Generate MiniLM embeddings.
   - Populate Weaviate with BM25 + dense hybrid.

3. **Retriever + Generator Integration**
   - Connect retriever to llama.cpp FastAPI backend.
   - Add citation formatting.

4. **UI Layer**
   - Streamlit interface with query, graph, and source tabs.

5. **Evaluation + Iteration**
   - Implement automatic retrieval metrics and qualitative tests.

## Long-Term Ideas
- Add new mythologies (Egyptian, Mesopotamian)
- Visual relationship graphs (NetworkX → Plotly)
- Local fine-tuning on mythological Q&A pairs
