
---

# 📌 **build_index.py**

```python
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os

# Load sample docs
docs = []
with open("docs/sample_docs.txt", "r", encoding="utf8") as f:
    for line in f:
        docs.append(line.strip())

# Create embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embs = model.encode(docs, convert_to_numpy=True)

# Build FAISS index
d = embs.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embs)

# Save index and docs
faiss.write_index(index, "docs/faiss.index")
json.dump(docs, open("docs/docs.json", "w", encoding="utf8"), ensure_ascii=False, indent=2)

print("FAISS index built successfully.")
