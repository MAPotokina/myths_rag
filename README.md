# MythRAG — Retrieval-Augmented Q&A for Greek Mythology

**One-liner:**  
Local-first RAG that answers myth questions with citations and maps Greek deities relationships.

**Why it’s interesting:**  
Hybrid retrieval (**Weaviate BM25 + dense**), graph reasoning (**NetworkX**), runs fully offline on a **CPU-only laptop**.

---

### 🎯 Demo Goals

**Ask:**  
“Who are the parents of Perseus?” → cited passages (*Apollodorus*).  

**Graph:**  
Shortest path between **Athena** and **Perseus**.  

**Compare:**  
**Zeus ↔ Odin** with aligned sources (*Apollodorus* vs *Prose Edda*).

---

### ⚙️ Stack (Local-First)

- **Weaviate** (hybrid search)
- **MiniLM** embeddings (CPU)
- **llama.cpp** (1–3 B GGUF models)
- **FastAPI**
- **Streamlit**
- **NetworkX**

---

### 🧩 Status
Scaffolding in progress; ingestion + hybrid search next.

---

### 🚀 How to Run (later)
```bash
docker compose up weaviate  # … TBD
```

---

### 📄 License & Sources
See LICENSE and docs/SOURCES.md

---
