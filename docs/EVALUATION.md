# Evaluation Plan

## 1. Retrieval Quality

### Quantitative
- **Recall@k / Precision@k:** test if correct passages appear in top-k results.
- **BM25 vs Hybrid:** compare pure lexical vs hybrid search on a small gold set.
- **Latency:** average query time on CPU-only setup.

### Qualitative
- Inspect 10 sample questions:
  - "Who is the mother of Perseus?"
  - "Who forged Thor’s hammer?"
  - "Compare Zeus and Odin."
- Check: relevance, citation correctness, duplication, mythology context accuracy.

---

## 2. Generation Quality
- Evaluate LLM responses for:
  - Citation inclusion (✅/❌)
  - Faithfulness to source passages
  - Mythology mix-ups (Greek/Norse cross-confusion)
- Use a simple 1–5 rating scale.

---

## 3. Graph Reasoning
- Verify NetworkX graph paths for correctness:
  - Shortest path between Athena ↔ Perseus.
  - Degree centrality for top Greek deities.
- Compare with textual relationships.

---

## 4. Offline Performance
- Measure end-to-end latency on CPU-only laptop.
- Target: < 3 s retrieval, < 8 s total generation for small models.

---

## 5. Future Evaluation Enhancements
- Add unit tests with synthetic queries.
- Incorporate mythological fact-checking dataset.
