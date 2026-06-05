import json
import re
from pathlib import Path

# Resolve absolute paths
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
WORKSPACE_DIR = BACKEND_DIR.parent
INPUT_PATH = WORKSPACE_DIR / "edmentor_FINAL_FINETUNE.jsonl"
OUTPUT_PATH = WORKSPACE_DIR / "rag_docs.json"

print(f"Reading dataset from: {INPUT_PATH}")
print(f"Output path: {OUTPUT_PATH}")

clean_docs = []
seen = set()

# Open file with UTF-8 encoding to avoid Windows encoding errors
with open(INPUT_PATH, "r", encoding="utf-8") as infile:
    for line in infile:
        if not line.strip():
            continue
        d = json.loads(line)
        msgs = d["messages"]
        user_q = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst_a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        
        # Deduplicate using user question prefix
        key = user_q[:80]
        if key in seen:
            continue
        seen.add(key)
        
        # Drop markdown contamination
        if any(c in asst_a for c in ["**", "##", "\n-", "\n*", "\n1."]):
            continue
        
        # Drop off-domain (code-writing, non-mentor tasks)
        bad_signals = ["write a python", "write code", "dll", "go build", 
                       "tailwind css", "modify the skill map"]
        if any(s in user_q.lower() for s in bad_signals):
            continue
        
        # Drop responses with broken formatting (leftover punctuation artifacts)
        if re.search(r'\b(intinput|printodds|odds\.appendi)\b', asst_a):
            continue
        
        # Length filter: drop too-short or too-long
        if not (50 < len(asst_a) < 750):
            continue
        
        clean_docs.append({
            "question": user_q,
            "answer": asst_a,
            # Combined for embedding — question gives semantic context
            "text": f"Student: {user_q}\nMentor: {asst_a}"
        })

print(f"Clean docs for RAG: {len(clean_docs)}")

# Save to output file
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(clean_docs, f, indent=2)

print("Dataset cleaning completed successfully.")
