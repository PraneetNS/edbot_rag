import json
import re
import logging
from pathlib import Path
from llama_index.core import Document
from typing import List

logger = logging.getLogger(__name__)

# ── Topic inference keywords (used when 'topic' field is absent in the JSONL) ─
_TOPIC_KEYWORDS = {
    "dsa":          ["dsa", "algorithm", "data structure", "tree", "graph", "sort", "dynamic programming", "recursion"],
    "placement":    ["placement", "internship", "job", "interview", "resume", "recruit", "offer"],
    "cloud":        ["cloud", "aws", "gcp", "azure", "ec2", "s3", "deployment"],
    "frontend":     ["frontend", "react", "html", "css", "javascript", "web dev"],
    "backend":      ["backend", "api", "server", "database", "sql", "django", "flask", "node"],
    "ml":           ["machine learning", "ml", "deep learning", "nlp", "ai", "neural"],
    "programming":  ["python", "java", "c++", "golang", "rust", "code", "programming", "language"],
    "career":       ["career", "roadmap", "switch", "path", "skills", "goal", "mentor"],
    "productivity": ["procrastinat", "motivation", "focus", "habit", "routine", "balance", "study"],
    "open source":  ["open source", "github", "contribution", "pr", "pull request"],
}

def _infer_topic(text: str) -> str:
    """Return the most likely topic label based on keyword presence."""
    text_lower = text.lower()
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return topic.title()
    return "General"

def _strip_id_suffix(question: str) -> str:
    """
    Strip synthetic duplicate index tags like '[1446]' from the end of a question.
    Example: 'should i learn cloud during college [1446]' → 'should i learn cloud during college'
    """
    return re.sub(r"\s*\[\d+\]\s*$", "", question).strip()


def is_educational_query(query: str) -> bool:
    """Check if the user query is within the educational / mentoring domain."""
    query_lower = query.lower().strip()
    
    # Immediately block unsafe or explicitly out-of-scope categories
    safety_blocklist = [
        r"\bhack(ing|er|s)?\b", r"\bexploits?\b", r"\bbypass\s+security\b", r"\bpolitics?\b", 
        r"\belections?\b", r"\bspam\b", r"\brecipes?\b", r"\bmovies?\b", r"\bgames?\b", 
        r"\bbake\s+a\s+cake\b", r"\btell\s+a\s+joke\b", r"\bweather\b"
    ]
    for pattern in safety_blocklist:
        if re.search(pattern, query_lower):
            return False
            
    educational_keywords = [
        "course", "placement", "internship", "certif", "lms", "support", "help",
        "contact", "learn", "study", "exam", "admission", "fee", "price", "duration",
        "react", "python", "java", "coding", "partner", "vtu", "job", "career",
        "syllabus", "curriculum", "schedule", "class", "project", "assignment", "bot", 
        "hello", "hi", "hey", "edutainer", "edubot", "who are you", "what is your name", "help",
        "what", "how", "why", "who", "where", "when", "can", "explain", "tell", "describe",
        "programming", "code", "developer", "development", "software", "database", "sql",
        "computer", "science", "engineering", "math", "algorithm", "data", "structure",
        "learn", "study", "teach", "explain", "tutorial", "guide", "concept", "javascript",
        "c++", "c#", "ruby", "rust", "go", "php", "web", "html", "css", "git", "github",
        "frontend", "backend", "fullstack", "technology", "network", "security",
        "2nd year", "second year", "3rd year", "third year", "4th year", "fourth year",
        "final year", "sophomore", "junior", "senior", "capstone", "dsa", "oop", 
        "system design", "resume", "interview", "mentor", "roadmaps", "placement prep"
    ]
    
    if any(keyword in query_lower for keyword in educational_keywords):
        return True
    return False


def load_jsonl_conversations(file_path: str, deduplicate: bool = False, dedup_threshold: float = 0.92) -> List[Document]:
    """
    Reads the JSONL dataset, validates conversations, filters out system messages,
    extracts student questions and mentor responses, and returns them as formatted
    educational documents.

    Args:
        file_path:       Absolute path to the .jsonl file.
        deduplicate:     If True, performs semantic cosine similarity deduplication.
        dedup_threshold: Cosine similarity threshold for semantic deduplication (default: 0.92).
    """
    documents = []
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset JSONL not found at: {file_path}")

    classifier = None

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                messages = data.get("messages", [])

                # Gracefully handle missing topic/source (common in the 5k file)
                raw_topic  = data.get("topic", "").strip()
                source     = data.get("source", "Edumentor Dataset").strip()

                if not isinstance(messages, list):
                    continue

                user_questions    = []
                assistant_answers = []

                for msg in messages:
                    role    = msg.get("role")
                    content = msg.get("content", "").strip()
                    if not content:
                        continue
                    if role == "user":
                        user_questions.append(content)
                    elif role == "assistant":
                        assistant_answers.append(content)

                if not user_questions or not assistant_answers:
                    continue   # skip empty / malformed conversations

                student_question   = "\n".join(user_questions)
                mentor_explanation = "\n".join(assistant_answers)

                # Strip synthetic duplicate index tags like '[1446]'
                student_question = _strip_id_suffix(student_question)

                # Gate: Filter out non-educational / out-of-scope samples
                if not is_educational_query(student_question):
                    continue

                # Infer topic when field is absent using the centroid classifier
                if not raw_topic and classifier is None:
                    from edmentor.topic_classifier import EdmentorTopicClassifier
                    classifier = EdmentorTopicClassifier()

                topic = raw_topic if raw_topic else (
                    classifier.classify(student_question) if classifier else "General"
                )

                # Format as conversational mentor chunk (spec: Student/Mentor voice format)
                formatted_text = (
                    f"Student: {student_question}\n"
                    f"Mentor: {mentor_explanation}"
                )

                doc = Document(
                    text=formatted_text,
                    metadata={
                        "topic":             topic,
                        "source":            source,
                        "original_question": student_question,
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.warning(f"Error parsing line {idx}: {e}")

    if deduplicate and documents:
        import numpy as np
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        questions = [doc.metadata["original_question"] for doc in documents]
        logger.info(f"Generating embeddings for {len(questions)} questions for semantic deduplication...")
        embeddings = embed_model.get_text_embedding_batch(questions, show_progress=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalise embeddings for dot product cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_embeddings = embeddings / norms

        accepted_docs = []
        accepted_embs = []

        for idx, doc in enumerate(documents):
            emb = norm_embeddings[idx]
            if not accepted_docs:
                accepted_docs.append(doc)
                accepted_embs.append(emb)
                continue

            # Compute cosine similarities vs all accepted
            sims = np.dot(accepted_embs, emb)
            max_sim = np.max(sims)
            if max_sim <= dedup_threshold:
                accepted_docs.append(doc)
                accepted_embs.append(emb)

        logger.info(f"Semantic deduplication: loaded {len(accepted_docs)} unique documents out of {len(documents)} total (threshold={dedup_threshold}).")
        documents = accepted_docs
    else:
        logger.info(f"Loaded {len(documents)} clean mentoring documents (deduplication OFF).")

    return documents


