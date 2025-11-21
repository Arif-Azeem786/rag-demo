import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Load index + docs
index = faiss.read_index("docs/faiss.index")
docs = json.load(open("docs/docs.json", "r", encoding="utf8"))

def retrieve(query, k=3):
    q_emb = model.encode([query])[0].astype('float32')
    D, I = index.search(np.array([q_emb]), k)
    return [docs[int(i)] for i in I[0]]

def answer_with_llm(question, contexts):
    prompt = f"Use the following contexts to answer:\n{contexts}\nQuestion: {question}\nAnswer:"
    resp = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return resp.choices[0].text.strip()

if __name__ == "__main__":
    import sys
    query = sys.argv[1]

    retrieved = retrieve(query)
    print("\nRetrieved:")
    for r in retrieved:
        print("-", r)

    print("\nFinal Answer:")
    if openai.api_key:
        print(answer_with_llm(query, retrieved))
    else:
        print("OpenAI key not found. Showing retrieved chunks only.")
