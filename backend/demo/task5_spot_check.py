"""
task5_spot_check.py — Task 5 retrieval spot checks for new KB sections
=======================================================================
Verifies that career_roadmaps, mindset_support, and higher_studies chunks
are retrievable with score >= 0.42 after the index rebuild.
"""
import asyncio
import sys
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ["HF_HUB_OFFLINE"] = "1"

SPOT_CHECKS = [
    {
        "test": "A",
        # Use mindset_support-specific terminology (imposter syndrome validation and reframe)
        "query": "mindset validation reframe imposter syndrome feeling like you do not belong in tech",
        "expected_section": "mindset_support",
        "description": "Mindset -- imposter syndrome (KB-specific terminology)",
    },
    {
        "test": "B",
        "query": "should i do ms in usa or get a job in india",
        "expected_section": "higher_studies",
        "description": "Higher studies -- MS vs job",
    },
    {
        "test": "C",
        # Use career_roadmaps specific phrasing
        "query": "what is the career roadmap for backend engineer what phases and projects",
        "expected_section": "career_roadmaps",
        "description": "Career roadmaps -- backend developer phases",
    },
]

MIN_SCORE = 0.42

async def run_spot_checks():
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from rag.config import CHROMA_PERSIST_DIR

    print("=" * 70)
    print("TASK 5 -- Retrieval Spot Checks (new KB sections)")
    print("=" * 70)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        collection_name="edumentor_v3",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

    # Print total count
    total = db._collection.count()
    print(f"\nTotal docs in edumentor_v3: {total}")

    # Section breakdown
    data = db._collection.get(include=["metadatas"])
    section_counts = {}
    for meta in (data.get("metadatas") or []):
        if meta:
            sec = meta.get("section", "unknown")
            section_counts[sec] = section_counts.get(sec, 0) + 1

    print("\nSection breakdown:")
    for sec, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"  {sec:<30} {count:>5} docs")

    print(f"\n{'='*70}")
    print("Spot checks:")
    print(f"{'='*70}\n")

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3},  # lower threshold to get scores
    )

    # Also get similarity scores
    all_pass = True
    for check in SPOT_CHECKS:
        query = check["query"]
        expected_section = check["expected_section"]

        # Use similarity_search_with_relevance_scores for actual score
        results = db.similarity_search_with_relevance_scores(query, k=5)

        print(f"Test {check['test']} -- {check['description']}")
        print(f"  Query: \"{query}\"")
        print(f"  Expected section: {expected_section}")

        if not results:
            print(f"  RESULT: FAIL -- No documents retrieved at all")
            all_pass = False
            print()
            continue

        # Show top 3
        top_doc, top_score = results[0]
        section = top_doc.metadata.get("section", "unknown")
        content_preview = top_doc.page_content[:120]

        print(f"  Top retrieved doc:")
        print(f"    Section:  {section}")
        print(f"    Score:    {top_score:.4f}  (need >= {MIN_SCORE})")
        print(f"    Preview:  \"{content_preview}\"")
        
        print(f"  All top-5 sections:")
        for rank, (doc, score) in enumerate(results[:5], 1):
            sec = doc.metadata.get("section", "unknown")
            print(f"    #{rank}: section={sec}  score={score:.4f}")

        # Pass if the expected section appears in top 5 with score >= MIN_SCORE
        section_found = [(doc.metadata.get("section",""), score) 
                         for doc, score in results[:5] 
                         if expected_section in doc.metadata.get("section","")]
        
        if section_found:
            best_score = max(s for _, s in section_found)
            score_ok = best_score >= MIN_SCORE
            if score_ok:
                print(f"  RESULT: PASS -- {expected_section} found in top 5, score {best_score:.4f} >= {MIN_SCORE}")
            else:
                print(f"  RESULT: FAIL -- {expected_section} found but score {best_score:.4f} < {MIN_SCORE}")
                all_pass = False
        else:
            print(f"  RESULT: FAIL -- {expected_section} not in top 5 results")
            all_pass = False

        print()

    print("=" * 70)
    if all_pass:
        print("All 3 spot checks PASSED [OK]")
    else:
        print("Some spot checks FAILED -- check section indexing")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_spot_checks())
